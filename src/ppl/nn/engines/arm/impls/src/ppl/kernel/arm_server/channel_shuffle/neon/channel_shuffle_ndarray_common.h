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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_NDARRAY_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_NDARRAY_COMMON_H_

#include <string.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode channel_shuffle_ndarray_concat_split_common(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst0_shape,
    const ppl::common::TensorShape *dst1_shape,
    const eT *src0,
    const eT *src1,
    const int32_t group,
    eT *dst0,
    eT *dst1)
{
    const int64_t src_c0   = src0_shape->GetDim(1);
    const int64_t src_c1   = src1_shape->GetDim(1);
    const int64_t channels = src_c0 + src_c1;
    const int64_t dst_c0   = dst0_shape->GetDim(1);
    const int64_t dst_c1   = dst1_shape->GetDim(1);

    if (channels % group != 0 || dst_c0 + dst_c1 != channels) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src0_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src0_shape->GetDimCount(); i++) {
        inner_dims *= src0_shape->GetDim(i);
    }

    const int64_t channels_per_group = channels / group;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t c = 0; c < channels_per_group; c++) {
            for (int64_t g = 0; g < group; g++) {
                const int64_t src_c_idx = g * channels_per_group + c;
                const int64_t dst_c_idx = c * group + g;

                const eT *p_src = src_c_idx < src_c0 ? src0 + (n * src_c0 + src_c_idx) * inner_dims : src1 + (n * src_c1 + src_c_idx - src_c0) * inner_dims;
                eT *p_dst       = dst_c_idx < dst_c0 ? dst0 + (n * dst_c0 + dst_c_idx) * inner_dims : dst1 + (n * dst_c1 + dst_c_idx - dst_c0) * inner_dims;

                memcpy(p_dst, p_src, inner_dims * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
static ppl::common::RetCode channel_shuffle_ndarray_concat_common(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    const int32_t group,
    eT *dst)
{
    const int64_t src_c0   = src0_shape->GetDim(1);
    const int64_t src_c1   = src1_shape->GetDim(1);
    const int64_t channels = src_c0 + src_c1;
    const int64_t dst_c    = dst_shape->GetDim(1);

    if (channels % group != 0 || dst_c != channels) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src0_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src0_shape->GetDimCount(); i++) {
        inner_dims *= src0_shape->GetDim(i);
    }

    const int64_t channels_per_group = channels / group;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t c = 0; c < channels_per_group; c++) {
            for (int64_t g = 0; g < group; g++) {
                const int64_t src_c_idx = g * channels_per_group + c;
                const int64_t dst_c_idx = c * group + g;

                const eT *p_src = src_c_idx < src_c0 ? src0 + (n * src_c0 + src_c_idx) * inner_dims : src1 + (n * src_c1 + src_c_idx - src_c0) * inner_dims;
                eT *p_dst       = dst + (n * dst_c + dst_c_idx) * inner_dims;

                memcpy(p_dst, p_src, inner_dims * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
static ppl::common::RetCode channel_shuffle_ndarray_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int32_t group,
    eT *dst)
{
    const int64_t channels = src_shape->GetDim(1);
    if (channels % group != 0 || channels != dst_shape->GetDim(1)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src_shape->GetDimCount(); i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    const int64_t channels_per_group = channels / group;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t c = 0; c < channels_per_group; c++) {
            for (int64_t g = 0; g < group; g++) {
                const int64_t src_c_idx = g * channels_per_group + c;
                const int64_t dst_c_idx = c * group + g;

                const eT *p_src = src + (n * channels + src_c_idx) * inner_dims;
                eT *p_dst       = dst + (n * channels + dst_c_idx) * inner_dims;

                memcpy(p_dst, p_src, inner_dims * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}} // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_NDARRAY_COMMON_H_