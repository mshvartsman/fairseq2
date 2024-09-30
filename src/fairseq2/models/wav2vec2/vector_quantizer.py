# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2.nn import Linear
from fairseq2.typing import DataType, Device


class VectorQuantizer(Module, ABC):
    """Quantizes incoming data in a differentiable way."""

    input_dim: int
    output_dim: int
    num_codebooks: int
    num_codebook_entries: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> "VectorQuantizerOutput":
        pass


@dataclass
class VectorQuantizerOutput(ABC):
    """Holds the output of a vector quantizer."""

    quantized_vectors: Tensor
    """The quantized vector output."""

    @abstractmethod
    def compute_loss(self) -> Tensor:
        """Compute the loss."""

    @abstractmethod
    def get_target_indices(self, num_codebooks: int) -> Tensor:
        pass


def renyi_entropy(p, alpha):
    # limits
    if alpha == 1:  # shannon entropy
        return -torch.sum(p * torch.log(p + 1e-7), dim=-1)
    # elif (alpha == 0): # hartley entropy, but who cares
    #     return torch.ones(len(p)) * log(len(p))
    # elif (alpha == torch.inf): # min-entropy (not differentiable)
    #     return -torch.log(torch.max(p))
    # regular
    else:
        return torch.log(torch.sum(p**alpha, dim=-1)) / (1 - alpha)


@final
class GumbelVectorQuantizer(VectorQuantizer):
    """Quantizes incoming data using Gumbel-Softmax."""

    input_dim: int
    output_dim: int
    num_codebooks: int
    num_codebook_entries: int
    min_temp: float
    max_temp: float
    temp_decay: float
    entry_proj: Linear
    entries: Parameter
    num_updates: Tensor

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebooks: int,
        num_codebook_entries: int,
        *,
        codebook_sampling_temperature: tuple[float, float, float],
        renyi_alpha: float = 1.0,  # shannon entropy by default
        use_perplexity: bool = True,
        use_uniform_penalty: bool = False,
        use_mlp: bool = False,
        rand_source: str = "gumbel",
        nonlinearity: str = "softmax",
        device: Device | None = None,
        dtype: DataType | None = None,
    ):
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        :param num_codebooks:
            number of groups for vector quantization
        :param num_codebook_entries:
            number of quantized vectors per group
        :param codebook_sampling_temperature:
            The temperature for training. A tuple of maximum temperature,
            minimum temperature, and decay factor.
        """
        super().__init__(input_dim, output_dim)

        if output_dim % num_codebooks != 0:
            raise ValueError(
                f"`output_dim` must be a multiple of `num_codebooks` ({num_codebooks}), but is {output_dim} instead."
            )

        entry_dim = output_dim // num_codebooks

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_codebooks = num_codebooks
        self.num_codebook_entries = num_codebook_entries
        self.max_temp, self.min_temp, self.temp_decay = codebook_sampling_temperature
        self.renyi_alpha = renyi_alpha
        self.use_perplexity = use_perplexity
        self.use_uniform_penalty = use_uniform_penalty
        self.use_mlp = use_mlp
        self.rand_source = rand_source
        self.nonlinearity = nonlinearity

        num_total_entries = num_codebooks * num_codebook_entries

        if self.use_mlp:
            self.entry_proj = nn.Sequential(
                    Linear(
                        self.input_dim,
                        num_total_entries,
                        bias=True,
                        init_fn=init_entry_projection,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.ReLU(),
                    Linear(
                        num_total_entries,
                        num_total_entries,
                        bias=True,
                        device=device,
                        dtype=dtype,
                    ),
            )
            if self.rand_source == "bias":
                self.entry_proj[-1].bias.requires_grad_(False)

        else:
            self.entry_proj = Linear(
                self.input_dim,
                num_total_entries,
                bias=True,
                init_fn=init_entry_projection,
                device=device,
                dtype=dtype,
            )
            if self.rand_source == "bias":
                self.entry_proj.bias.requires_grad_(False)

        
            

        self.entries = Parameter(
            torch.empty((1, num_total_entries, entry_dim), device=device, dtype=dtype)
        )

        num_updates = torch.empty((), device=device, dtype=torch.int64)

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.entries)

        self.num_updates.zero_()

    @override
    def forward(self, x: Tensor) -> "GumbelVectorQuantizerOutput":
        current_temp = self._compute_current_temp()

        bsz, tsz, fsz = x.shape

        if self.rand_source == "bias":
            if isinstance(self.entry_proj, nn.Sequential):
                self.entry_proj[-1].bias.exponential_().log_()
            else: 
                self.entry_proj.bias.exponential_().log_()
    
        x = self.entry_proj(x)

        #        x = x.unflatten(-1, (self.num_codebooks, self.num_codebook_entries))
        #
        #        k = x.argmax(-1, keepdim=True)
        #
        #        hard_x = torch.zeros_like(x, dtype=torch.float32).scatter_(-1, k, 1.0)
        #
        #        hard_probs = hard_x.mean(dim=0)
        x = x.view(bsz * tsz * self.num_codebooks, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.num_codebooks, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)

        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.num_codebooks, -1).float(), dim=-1
        ).mean(dim=0)

        entropy = renyi_entropy(avg_probs + 1e-7, alpha=self.renyi_alpha)
        if self.use_perplexity:
            prob_perplexity = torch.exp(entropy).sum()
        else:
            prob_perplexity = entropy.sum()  # HACK this isn't perplexithy anymore

        # HACK just add it to perplexity for now
        if self.use_uniform_penalty:
            n = self.num_codebooks * self.num_codebook_entries
            uniform_perplexity = renyi_entropy(
                torch.abs(avg_probs - 1 / n) + 1e-7, alpha=self.renyi_alpha
            )
            if self.use_perplexity:
                prob_perplexity += torch.exp(uniform_perplexity).sum()
            else:
                prob_perplexity += uniform_perplexity.sum()

        # prob_perplexity = torch.exp(
        #     -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        # ).sum()
                # if self.rand_source == "gumbel":
                #     # baseline
                #     x = gumbel_softmax(x.float(), tau=current_temp, hard=True).type_as(x)
        if self.training:
            if self.rand_source == "bias": 
                # randomize the bias (above) instead of inside gumbel_softmax
                logits = x.float() / current_temp
            elif self.rand_source == "gumbel":
                # randomize with gumbel
                gumbels = -torch.empty_like(x.float()).exponential_().log()  # ~Gumbel(0,1)
                logits = (x.float() + gumbels) / current_temp  # ~Gumbel(logits,tau)
            else: 
                raise NotImplementedError()
            if self.nonlinearity == "softmax":
                soft_x = nn.functional.softmax(logits, dim=-1)
                hard_x = hard_x.reshape(*soft_x.shape) # already constructed above
            elif self.nonlinearity == "sigmoid":
                soft_x = nn.functional.sigmoid(logits)
                # since we don't normalize to sum to 1, hard_x isn't 
                # max, it's p > 0.5
                hard_x = (soft_x > 0.5).int()
            # straight-through gradient trick
            x = hard_x - soft_x.detach() + soft_x
        else: 
            x = hard_x
        x = x.view(bsz * tsz, -1)

        cb = x

        x = x.unsqueeze(-1) * self.entries
        x = x.view(bsz * tsz, self.num_codebooks, self.num_codebook_entries, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        return GumbelVectorQuantizerOutput(
            x,
            cb,
            self.num_codebooks,
            self.num_codebook_entries,
            code_perplexity,
            prob_perplexity,
            current_temp,
        )

    def _compute_current_temp(self) -> float:
        temp = self.max_temp * self.temp_decay ** int(self.num_updates)

        if self.training:
            self.num_updates.add_(1)

        return max(temp, self.min_temp)


def init_entry_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, mean=0.0, std=1.0)

    assert proj.bias is not None

    nn.init.zeros_(proj.bias)


@final
@dataclass
class GumbelVectorQuantizerOutput(VectorQuantizerOutput):
    cb: Tensor
    num_codebooks: int
    num_codebook_entries: int
    code_perplexity: Tensor
    prob_perplexity: Tensor
    temperature: float

    @override
    def compute_loss(self) -> Tensor:
        num_entries = self.num_codebooks * self.num_codebook_entries

        return (num_entries - self.prob_perplexity) / num_entries  # type: ignore[no-any-return]

    @override
    def get_target_indices(self, num_codebooks: int) -> Tensor:
        batch_size, seq_len = self.quantized_vectors.shape[:2]

        cb = self.cb.view(batch_size * seq_len * self.num_codebooks, -1)

        indices = cb.argmax(dim=-1).view(-1, self.num_codebooks)

        indices = indices[..., :num_codebooks]

        return indices.detach()
