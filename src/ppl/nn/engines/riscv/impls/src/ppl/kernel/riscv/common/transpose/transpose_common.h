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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_

#include <cstring>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename T>
ppl::common::RetCode transpose2d_ndarray(
    const T* src,
    T* dst,
    const ppl::common::TensorShape* src_shape)
{
    const int32_t src_h = src_shape->GetDim(0);
    const int32_t src_w = src_shape->GetDim(1);

    for (int32_t i = 0; i < src_h; ++i) {
        for (int32_t j = 0; j < src_w; ++j) {
            dst[j * src_h + i] = src[i * src_w + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode transpose3d_ndarray(
    const T* src,
    T* dst,

    const int32_t* perm,
    const ppl::common::TensorShape* src_shape)
{
    const int32_t channels = src_shape->GetDim(0);
    const int32_t src_h    = src_shape->GetDim(1);
    const int32_t src_w    = src_shape->GetDim(2);
    int64_t src_dims[3]    = {channels, src_h, src_w};

    int32_t dst_channels = src_dims[perm[0]];
    int32_t dst_src_h    = src_dims[perm[1]];
    int32_t dst_src_w    = src_dims[perm[2]];

    int64_t src_stride[3] = {int64_t(src_h) * src_w, src_w, 1};
    int64_t dst_stride[3] = {int64_t(dst_src_h) * dst_src_w, dst_src_w, 1};
    int64_t axis_stride[3];
    for (int32_t i = 0; i < 3; ++i) {
        axis_stride[perm[i]] = dst_stride[i];
    }

    for (int32_t c = 0; c < channels; ++c) {
        int64_t channels_in_offset  = c * src_stride[0];
        int64_t channels_out_offset = c * axis_stride[0];
        for (int32_t h = 0; h < src_h; ++h) {
            int64_t height_in_offset  = h * src_stride[1];
            int64_t height_out_offset = h * axis_stride[1];
            for (int32_t w = 0; w < src_w; ++w) {
                dst[channels_out_offset + height_out_offset + w * axis_stride[2]] =
                    src[channels_in_offset + height_in_offset + w * 1];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode transpose4d_ndarray(
    const T* src,
    T* dst,

    const int32_t* perm,
    const ppl::common::TensorShape* src_shape)
{
    const int32_t batch    = src_shape->GetDim(0);
    const int32_t channels = src_shape->GetDim(1);
    const int32_t src_h    = src_shape->GetDim(2);
    const int32_t src_w    = src_shape->GetDim(3);
    int64_t src_dims[4]    = {batch, channels, src_h, src_w};

    int32_t dst_batch    = src_dims[perm[0]];
    int32_t dst_channels = src_dims[perm[1]];
    int32_t dst_src_h    = src_dims[perm[2]];
    int32_t dst_src_w    = src_dims[perm[3]];

    int64_t src_stride[4] = {int64_t(channels) * src_h * src_w, int64_t(src_h) * src_w, src_w, 1};
    int64_t dst_stride[4] = {int64_t(dst_channels) * dst_src_h * dst_src_w, int64_t(dst_src_h) * dst_src_w, dst_src_w, 1};
    int64_t axis_stride[4];
    for (int32_t i = 0; i < 4; ++i) {
        axis_stride[perm[i]] = dst_stride[i];
    }

    for (int64_t n = 0; n < batch; ++n) {
        int64_t batch_in_offset  = n * src_stride[0];
        int64_t batch_out_offset = n * axis_stride[0];
        for (int64_t c = 0; c < channels; ++c) {
            int64_t channels_in_offset  = c * src_stride[1];
            int64_t channels_out_offset = c * axis_stride[1];
            for (int64_t h = 0; h < src_h; ++h) {
                int64_t height_in_offset  = h * src_stride[2];
                int64_t height_out_offset = h * axis_stride[2];
                int64_t base_in_offset    = batch_in_offset + channels_in_offset + height_in_offset;
                int64_t base_out_offset   = batch_out_offset + channels_out_offset + height_out_offset;
                for (int64_t w = 0; w < src_w; ++w) {
                    dst[base_out_offset + w * axis_stride[3]] = src[base_in_offset + w * 1];
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
void transpose_ndarray_recursive(
    const int64_t* src_dims,
    const int64_t* src_stride,
    const int64_t* dst_stride,
    const int32_t* perm,
    const uint32_t dim_idx,
    const uint32_t dim_count,
    const int64_t base_in_offset,
    const int64_t base_out_offset,

    const T* src,
    T* dst)
{
    const int64_t length = src_dims[dim_idx];
    if (dim_idx == dim_count - 1) {
        for (int64_t i = 0; i < length; i++) {
            dst[base_out_offset + i * dst_stride[perm[dim_idx]]] = src[base_in_offset + i];
        }
    } else {
        for (int64_t i = 0; i < length; i++) {
            transpose_ndarray_recursive<T>(
                src_dims,
                src_stride,
                dst_stride,
                perm,
                dim_idx,
                dim_count,
                base_in_offset,
                base_out_offset,
                src,
                dst);
        }
    }
}

template <typename T>
ppl::common::RetCode transpose_ndarray(
    const T* src,
    T* dst,

    const int32_t* perm,
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape)
{
    const uint32_t dim_count = src_shape->GetDimCount();

    if (dim_count > PPL_RISCV_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (dim_count <= 1) {
        return ppl::common::RC_SUCCESS;
    }
    if (dim_count <= 2) {
        return transpose2d_ndarray(src, dst, src_shape);
    }
    if (dim_count <= 3) {
        return transpose3d_ndarray(src, dst, perm, src_shape);
    }
    if (dim_count <= 4) {
        return transpose4d_ndarray(src, dst, perm, src_shape);
    }

    auto src_dims                                   = src_shape->GetDims();
    auto dst_dims                                   = dst_shape->GetDims();
    int64_t src_stride[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t dst_stride[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    src_stride[dim_count - 1]                       = 1;
    dst_stride[dim_count - 1]                       = 1;
    for (int32_t i = (int32_t)dim_count - 2; i >= 0; i--) {
        src_stride[i] = src_stride[i + 1] * src_dims[i + 1];
        dst_stride[i] = dst_stride[i + 1] * dst_dims[i + 1];
    }

    int32_t revert_perm[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    for (int64_t i = 0; i < dim_count; i++) {
        const int32_t perm_val = perm[i] < 0 ? perm[i] + dim_count : perm[i];
        revert_perm[perm_val]  = i;
    }

    transpose_ndarray_recursive(src_dims, src_stride, dst_stride, revert_perm, 0, dim_count, 0, 0, src, dst);

    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode transpose_ndarray_continous2d(
    const T* src,
    T* dst,

    const ppl::common::TensorShape* src_shape,
    const uint32_t axis0,
    const uint32_t axis1)
{
    const uint32_t dim_count = src_shape->GetDimCount();
    const int64_t dim0       = src_shape->GetDim(axis0);
    const int64_t dim1       = src_shape->GetDim(axis1);

    int64_t outer_dims = 1;
    for (uint32_t i = 0; i < axis0; i++) {
        outer_dims *= src_shape->GetDim(i);
    }

    int64_t inner_dims = 1;
    for (uint32_t i = axis1 + 1; i < dim_count; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    for (int64_t od = 0; od < outer_dims; od++) {
        for (int64_t d0 = 0; d0 < dim0; d0++) {
            for (int64_t d1 = 0; d1 < dim1; d1++) {
                const T* src_ = src + od * dim0 * dim1 * inner_dims + d0 * dim1 * inner_dims + d1 * inner_dims;
                T* dst_       = dst + od * dim1 * dim0 * inner_dims + d1 * dim0 * inner_dims + d0 * inner_dims;
                memcpy(dst_, src_, inner_dims * sizeof(T));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_
