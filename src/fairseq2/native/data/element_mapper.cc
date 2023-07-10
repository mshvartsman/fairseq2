// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/element_mapper.h"

#include <exception>
#include <stdexcept>

#include "fairseq2/native/fmt.h"
#include "fairseq2/native/detail/exception.h"

using namespace fairseq2::detail;

namespace fairseq2 {

data
element_mapper::operator()(data &&d)
{
    if (!selector_)
        return map_fn_(std::move(d));

    selector_->visit(d, [this](data &element, element_path_ref path)
    {
        try {
            element = map_fn_(std::move(element));
        } catch (const std::exception &) {
            throw_with_nested<std::runtime_error>(
                "The map function has failed while processing the path '{}' of the input data. See nested exception for details.", path);
        }
    });

    return std::move(d);
}

}  // namespace fairseq2