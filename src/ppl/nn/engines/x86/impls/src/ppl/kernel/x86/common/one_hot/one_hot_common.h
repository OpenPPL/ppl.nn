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

#ifndef __ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/common/log.h"
namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
static ppl::common::RetCode cast2int64_kernel(const eT *src, int64_t *&dst, const uint64_t length, int64_t depth_val)
{
    if (!dst) dst = (int64_t *)ppl::common::AlignedAlloc(length * sizeof(int64_t), 64);
    for (uint64_t i = 0; i < length; i++) {
        dst[i] = (int64_t)src[i];
    }
    return ppl::common::RC_SUCCESS;
}

static int64_t *cast2int64(const ppl::nn::TensorShape *src_shape, const void *src, int64_t depth_val)
{
    auto src_type = src_shape->GetDataType();
    if (src_type == ppl::common::DATATYPE_INT64) return (int64_t *)src;
    const uint64_t length = src_shape->CalcElementsExcludingPadding();
    int64_t *dst          = nullptr;
    ppl::common::RetCode ret;
    switch (src_type) {
        case ppl::common::DATATYPE_FLOAT64:
            ret = cast2int64_kernel<double>((double *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_FLOAT32:
            ret = cast2int64_kernel<float>((float *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_INT32:
            ret = cast2int64_kernel<int32_t>((int32_t *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_INT16:
            ret = cast2int64_kernel<int16_t>((int16_t *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_INT8:
            ret = cast2int64_kernel<int8_t>((int8_t *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_UINT32:
            ret = cast2int64_kernel<uint32_t>((uint32_t *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_UINT16:
            ret = cast2int64_kernel<uint16_t>((uint16_t *)src, dst, length, depth_val);
            break;
        case ppl::common::DATATYPE_UINT8:
            ret = cast2int64_kernel<uint8_t>((uint8_t *)src, dst, length, depth_val);
            break;
        default:
            ret = ppl::common::RC_INVALID_VALUE;
            break;
    }
    if (ret != ppl::common::RC_SUCCESS) return nullptr;
    return dst;
}

static int64_t cast2int64(const ppl::nn::TensorShape *src_shape, const void *src)
{
    auto src_type = src_shape->GetDataType();
    if (src_type == ppl::common::DATATYPE_INT64) return *((int64_t *)src);
    switch (src_type) {
        case ppl::common::DATATYPE_FLOAT64:
            return static_cast<int64_t>(*(double *)src);
        case ppl::common::DATATYPE_FLOAT32:
            return static_cast<int64_t>(*(float *)src);
        case ppl::common::DATATYPE_INT32:
            return static_cast<int64_t>(*(int32_t *)src);
        case ppl::common::DATATYPE_INT16:
            return static_cast<int64_t>(*(int16_t *)src);
        case ppl::common::DATATYPE_INT8:
            return static_cast<int64_t>(*(int8_t *)src);
        case ppl::common::DATATYPE_UINT32:
            return static_cast<int64_t>(*(uint32_t *)src);
        case ppl::common::DATATYPE_UINT16:
            return static_cast<int64_t>(*(uint16_t *)src);
        case ppl::common::DATATYPE_UINT8:
            return static_cast<int64_t>(*(uint8_t *)src);
        default:
            break;
    }
    return -1;
}

template <typename eT>
static ppl::common::RetCode one_hot_ndarray_common(
    const ppl::nn::TensorShape *indices_shape,
    const ppl::nn::TensorShape *depth_shape,
    const ppl::nn::TensorShape *values_shape,
    const void *indices,
    const void *depth,
    const eT *values,
    const int32_t axis,
    eT *dst)
{
    if (!indices || !depth || !values) return ppl::common::RC_INVALID_VALUE;
    if (values_shape->CalcElementsExcludingPadding() != 2) {
        LOG(ERROR) << "value tensor should be [off_value, on_value] ";
        return ppl::common::RC_INVALID_VALUE;
    }
    int64_t depth_val = cast2int64(depth_shape, depth);
    if (depth_val == -1) return ppl::common::RC_INVALID_VALUE;
    int64_t *real_indices = cast2int64(indices_shape, indices, depth_val);
    if (!real_indices) return ppl::common::RC_INVALID_VALUE;
    int32_t indices_rank = indices_shape->GetDimCount();
    int32_t real_axis    = axis < 0 ? axis + indices_rank + 1 : axis;
    if (real_axis < 0 || real_axis > indices_rank) {
        LOG(ERROR) << "axis param should be in range of [-rank(indices)-1, rank(indices)],  \
                      which is ["
                   << -indices_rank << ", " << indices_rank - 1 << "], but " << axis << " was found.";
        return ppl::common::RC_INVALID_VALUE;
    }
    int64_t outer_dim        = indices_shape->CalcElementsToDimensionExcludingPadding(real_axis);
    int64_t inner_dim        = indices_shape->CalcElementsFromDimensionExcludingPadding(real_axis);
    int64_t axis_dim         = depth_val;
    int64_t stride           = axis_dim * inner_dim;
    eT on_value              = values[1];
    eT off_value             = values[0];
    const int64_t l2_cap     = (ppl::common::GetCpuCacheL2() == 0) ? 256 * 1024 : ppl::common::GetCpuCacheL2();
    int64_t min_block_size   = (l2_cap / depth_val / sizeof(eT)) * 0.25;
    const int64_t num_block  = min<uint64_t>(PPL_OMP_MAX_THREADS(), div_up(outer_dim, min_block_size));
    const int64_t block_body = outer_dim / num_block;
    const int64_t block_tail = outer_dim % num_block;
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < num_block; i++) {
        const int64_t block_start = i * block_body + (block_tail > i ? i : block_tail);
        const int64_t block_size  = block_body + (block_tail > i ? 1 : 0);
        for (int64_t j = block_start; j < block_start + block_size; j++) {
            eT *dst_base = dst + j * axis_dim * inner_dim;
            std::fill(dst_base, dst_base + stride, off_value);
            for (int64_t k = 0; k < inner_dim; ++k) {
                int64_t idx = real_indices[j * inner_dim + k];
                // Any entries in the 'indices' input tensor with values outside the range [-depth, depth-1]
                // will result in one-hot representation with all 'off_value' values in the output tensor.
                if (idx < -depth_val || idx >= depth_val) continue;
                idx                    = idx < 0 ? idx + depth_val : idx;
                eT *p_dst              = dst_base + k;
                p_dst[idx * inner_dim] = on_value;
            }
        }
    }
    if (real_indices != indices) ppl::common::AlignedFree(real_indices);
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_
