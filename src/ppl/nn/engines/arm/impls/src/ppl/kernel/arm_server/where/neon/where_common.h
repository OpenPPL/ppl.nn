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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_WHERE_NEON_WHERE_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_WHERE_NEON_WHERE_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
ppl::common::RetCode where_eltwise_common(
    const ppl::common::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    const uint32_t length = dst_shape->CalcElementsIncludingPadding();

    PRAGMA_OMP_PARALLEL_FOR()
    for (uint32_t i = 0; i < length; i++) {
        dst[i] = cond[i] != 0 ? src0[i] : src1[i];
    }

    return ppl::common::RC_SUCCESS;    
}

template <typename eT>
ppl::common::RetCode where_ndarray_recursive(
    const ppl::common::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src0,
    const eT *src1,
    const int64_t *inc_cond,
    const int64_t *inc0,
    const int64_t *inc1,
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
                eT x_value       = src0[i * inc0[dim_idx]];
                eT y_value       = src1[i * inc1[dim_idx]];
                dst[i]          = cond_value ? x_value : y_value;
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                bool cond_value = cond[i * inc_cond[dim_idx]];
                eT x_value       = src0[i * inc0[dim_idx]];
                eT y_value       = src1[i * inc1[dim_idx]];
                dst[i]          = cond_value ? x_value : y_value;
            }
        }
    } else {
        if (length > 1 && !has_paralleld) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                where_ndarray_recursive<eT>(dst_shape, cond + i * inc_cond[dim_idx], src0 + i * inc0[dim_idx], src1 + i * inc1[dim_idx], inc_cond, inc0, inc1, inc_out, dim_idx + 1, true, dst + i * inc_out[dim_idx]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                where_ndarray_recursive<eT>(dst_shape, cond + i * inc_cond[dim_idx], src0 + i * inc0[dim_idx], src1 + i * inc1[dim_idx], inc_cond, inc0, inc1, inc_out, dim_idx + 1, has_paralleld, dst + i * inc_out[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline void where_pad_shape(
    const ppl::common::TensorShape *shape,
    const int64_t padded_dim_count,
    int64_t* padded_shape)
{
    const int64_t dim_diff = padded_dim_count - shape->GetRealDimCount();
    for (int64_t i = 0; i < dim_diff; i++) {
        padded_shape[i] = 1;
    }
    for (int64_t i = dim_diff; i < padded_dim_count; i++) {
        padded_shape[i] = shape->GetDim(i - dim_diff);
    }
}

template <typename eT>
ppl::common::RetCode where_ndarray_common(
    const ppl::common::TensorShape *cond_shape,
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    // pad input dim
    const uint64_t max_dim_count             = dst_shape->GetDimCount();
    const uint64_t dims = 8;
    std::vector<int64_t> padded_cond_shape(max_dim_count);
    std::vector<int64_t> padded_src0_shape(max_dim_count);
    std::vector<int64_t> padded_src1_shape(max_dim_count);
    where_pad_shape(cond_shape, max_dim_count, padded_cond_shape.data());
    where_pad_shape(src0_shape, max_dim_count, padded_src0_shape.data());
    where_pad_shape(src1_shape, max_dim_count, padded_src1_shape.data());

    // prepare incs
    int64_t inc_cond[dims] = {0};
    int64_t inc0[dims]     = {0};
    int64_t inc1[dims]     = {0};
    int64_t inc_out[dims]  = {0};

    int64_t stride_cond = 1;
    int64_t stride_x    = 1;
    int64_t stride_y    = 1;
    int64_t stride_out  = 1;

    for (int32_t i = max_dim_count - 1; i >= 0; i--) {
        inc_cond[i] = padded_cond_shape[i] == 1 ? 0 : stride_cond;
        inc0[i]    = padded_src0_shape[i] == 1 ? 0 : stride_x;
        inc1[i]    = padded_src1_shape[i] == 1 ? 0 : stride_y;
        inc_out[i]  = stride_out;

        stride_cond *= padded_cond_shape[i];
        stride_x *= padded_src0_shape[i];
        stride_y *= padded_src1_shape[i];
        stride_out *= dst_shape->GetDim(i);
    }

    return where_ndarray_recursive<eT>(dst_shape, cond, src0, src1, inc_cond, inc0, inc1, inc_out, 0, false, dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_WHERE_NEON_WHERE_COMMON_H_
