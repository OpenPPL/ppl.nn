// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_PPL_KERNEL_X86_COMMON_CUMSUM_CUMSUM_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_CUMSUM_CUMSUM_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

template<typename eT, int64_t exclusive, int64_t reverse>
void cumsum_ndarray_impl(
    const eT *x,
    const int64_t outer_dim,
    const int64_t cumsum_dim,
    const int64_t inner_dim,
    eT *y)
{
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t od = 0; od < outer_dim; ++od) {
        for (int64_t id = 0; id < inner_dim; ++id) {
            const eT *cx = x + od * cumsum_dim * inner_dim + id;
            eT *cy = y + od * cumsum_dim * inner_dim + id;
            if (reverse) {
                cx += (cumsum_dim - 1) * inner_dim;
                cy += (cumsum_dim - 1) * inner_dim;
            }

            eT current_sum = static_cast<eT>(0);
            for (int64_t cd = 0; cd < cumsum_dim; ++cd) {
                if (exclusive) cy[0] = current_sum;
                current_sum += cx[0];
                if (!exclusive) cy[0] = current_sum;
                
                if (reverse) {
                    cx -= inner_dim;
                    cy -= inner_dim;
                } else {
                    cx += inner_dim;
                    cy += inner_dim;
                }
            }
        }
    }
}

template<typename eT>
ppl::common::RetCode cumsum_ndarray(
    const ppl::nn::TensorShape *x_shape,
    const eT *x,
    const int64_t axis,
    const int64_t exclusive,
    const int64_t reverse,
    eT *y)
{
    const int64_t dim_count = x_shape->GetDimCount();
    const int64_t real_axis = axis >= 0 ? axis : dim_count + axis;

    if (real_axis + 1 > dim_count || real_axis < 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t cumsum_dim = x_shape->GetDim(real_axis);
    int64_t outer_dim = 1;
    int64_t inner_dim = 1;

    for (int64_t i = 0; i < real_axis; ++i) {
        outer_dim *= x_shape->GetDim(i);
    }

    for (int64_t i = real_axis + 1; i < dim_count; ++i) {
        inner_dim *= x_shape->GetDim(i);
    }

    auto impl_func = cumsum_ndarray_impl<eT, true, true>;
    if (exclusive) {
        if (!reverse) impl_func = cumsum_ndarray_impl<eT, true, false>;
    } else {
        if (reverse) impl_func = cumsum_ndarray_impl<eT, false, true>;
        else impl_func = cumsum_ndarray_impl<eT, false, false>;
    }

    impl_func(x, outer_dim, cumsum_dim, inner_dim, y);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_CUMSUM_CUMSUM_COMMON_H_
