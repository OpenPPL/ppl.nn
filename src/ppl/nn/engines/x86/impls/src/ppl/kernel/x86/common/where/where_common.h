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

#ifndef __ST_PPL_KERNEL_X86_COMMON_WHERE_WHERE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_WHERE_WHERE_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode where_eltwise_common(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src_x,
    const eT *src_y,
    eT *dst)
{
    const uint32_t length = dst_shape->GetElementsIncludingPadding();

    PRAGMA_OMP_PARALLEL_FOR()
    for (uint32_t i = 0; i < length; i++) {
        dst[i] = cond[i] != 0 ? src_x[i] : src_y[i];
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode where_ndarray_recursive(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src_x,
    const eT *src_y,
    const int64_t *inc_cond,
    const int64_t *inc_x,
    const int64_t *inc_y,
    const int64_t *inc_out,
    const uint32_t dim_idx,
    const bool has_paralleld,
    eT *dst)
{
    const uint32_t dim_count = dst_shape->GetDimCount();
    const int64_t length    = dst_shape->GetDim(dim_idx);

    if (dim_idx == dim_count - 1) {
        if (length > 1 && !has_paralleld) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                bool cond_value = cond[i * inc_cond[dim_idx]];
                eT x_value       = src_x[i * inc_x[dim_idx]];
                eT y_value       = src_y[i * inc_y[dim_idx]];
                dst[i]          = cond_value ? x_value : y_value;
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                bool cond_value = cond[i * inc_cond[dim_idx]];
                eT x_value       = src_x[i * inc_x[dim_idx]];
                eT y_value       = src_y[i * inc_y[dim_idx]];
                dst[i]          = cond_value ? x_value : y_value;
            }
        }
    } else {
        if (length > 1 && !has_paralleld) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                where_ndarray_recursive<eT>(dst_shape, cond + i * inc_cond[dim_idx], src_x + i * inc_x[dim_idx], src_y + i * inc_y[dim_idx], inc_cond, inc_x, inc_y, inc_out, dim_idx + 1, true, dst + i * inc_out[dim_idx]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                where_ndarray_recursive<eT>(dst_shape, cond + i * inc_cond[dim_idx], src_x + i * inc_x[dim_idx], src_y + i * inc_y[dim_idx], inc_cond, inc_x, inc_y, inc_out, dim_idx + 1, has_paralleld, dst + i * inc_out[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline ppl::nn::TensorShape pad_shape(
    const ppl::nn::TensorShape *shape,
    const int32_t padded_dim_count)
{
    ppl::nn::TensorShape padded_shape(*shape);
    padded_shape.SetDimCount(padded_dim_count);
    if (shape->IsScalar()) {
        for (int32_t i = 0; i < padded_dim_count; i++) {
            padded_shape.SetDim(i, 1);
        }
    } else {
        const int32_t dim_diff = padded_dim_count - shape->GetDimCount();
        for (int32_t i = 0; i < dim_diff; i++) {
            padded_shape.SetDim(i, 1);
        }
        for (int32_t i = dim_diff; i < padded_dim_count; i++) {
            padded_shape.SetDim(i, shape->GetDim(i - dim_diff));
        }
    }
    return padded_shape;
}

template <typename eT>
ppl::common::RetCode where_ndarray_common(
    const ppl::nn::TensorShape *cond_shape,
    const ppl::nn::TensorShape *src_x_shape,
    const ppl::nn::TensorShape *src_y_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src_x,
    const eT *src_y,
    eT *dst)
{
    // pad input dim
    const uint32_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    ppl::nn::TensorShape padded_cond_shape = pad_shape(cond_shape, dim_count);
    ppl::nn::TensorShape padded_x_shape    = pad_shape(src_x_shape, dim_count);
    ppl::nn::TensorShape padded_y_shape    = pad_shape(src_y_shape, dim_count);

    // prepare incs
    int64_t inc_cond[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_x[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_y[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    int64_t stride_cond = 1;
    int64_t stride_x    = 1;
    int64_t stride_y    = 1;
    int64_t stride_out  = 1;

    for (uint32_t i = dim_count - 1; i >= 0; i--) {
        inc_cond[i] = padded_cond_shape.GetDim(i) == 1 ? 0 : stride_cond;
        inc_x[i]    = padded_x_shape.GetDim(i) == 1 ? 0 : stride_x;
        inc_y[i]    = padded_y_shape.GetDim(i) == 1 ? 0 : stride_y;
        inc_out[i]  = stride_out;

        stride_cond *= padded_cond_shape.GetDim(i);
        stride_x *= padded_x_shape.GetDim(i);
        stride_y *= padded_y_shape.GetDim(i);
        stride_out *= dst_shape->GetDim(i);
    }

    return where_ndarray_recursive<eT>(dst_shape, cond, src_x, src_y, inc_cond, inc_x, inc_y, inc_out, 0, false, dst);
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_WHERE_WHERE_COMMON_H_
