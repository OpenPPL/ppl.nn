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

#ifndef __ST_PPL_KERNEL_X86_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_

#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode transpose2d_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const eT *src,
    eT *dst)
{
    const int32_t src_h = src_shape->GetDim(0);
    const int32_t src_w = src_shape->GetDim(1);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int32_t i = 0; i < src_h; ++i) {
        for (int32_t j = 0; j < src_w; ++j) {
            dst[j * src_h + i] = src[i * src_w + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose3d_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const int32_t *perm,
    const eT *src,
    eT *dst)
{
    const int32_t channels                 = src_shape->GetDim(0);
    const int32_t src_h                    = src_shape->GetDim(1);
    const int32_t src_w                    = src_shape->GetDim(2);
    const int32_t src_axis_to_dst_channels = perm[0];
    const int32_t src_axis_to_dst_height   = perm[1];
    const int32_t src_axis_to_dst_width    = perm[2];
    int64_t src_dims[3]                    = {channels, src_h, src_w};
    int64_t src_stride[3]                  = {int64_t(src_h) * src_w, src_w, 1};
    int64_t axis_map[3]                    = {src_axis_to_dst_channels, src_axis_to_dst_height, src_axis_to_dst_width};
    int64_t dst_dims[3];
    for (int64_t i = 0; i < 3; ++i) {
        dst_dims[i] = src_dims[axis_map[i]];
    }
    int64_t dst_stride[3] = {dst_dims[1] * dst_dims[2], dst_dims[2], 1};
    int64_t axis_stride[3];
    for (int64_t i = 0; i < 3; ++i) {
        axis_stride[axis_map[i]] = dst_stride[i];
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int32_t c = 0; c < channels; ++c) {
        int64_t channel_in_offset  = c * src_stride[0];
        int64_t channel_out_offset = c * axis_stride[0];
        for (int64_t h = 0; h < src_h; ++h) {
            int64_t height_in_offset  = h * src_stride[1];
            int64_t height_out_offset = h * axis_stride[1];
            int64_t base_in_offset    = channel_in_offset + height_in_offset;
            int64_t base_out_offset   = channel_out_offset + height_out_offset;
            for (int64_t w = 0; w < src_w; ++w) {
                dst[base_out_offset + w * axis_stride[2]] = src[base_in_offset + w];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}


template <typename eT>
ppl::common::RetCode transpose4d_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const int32_t *perm,
    const eT *src,
    eT *dst)
{
    const int32_t batch                    = src_shape->GetDim(0);
    const int32_t channels                 = src_shape->GetDim(1);
    const int32_t src_h                    = src_shape->GetDim(2);
    const int32_t src_w                    = src_shape->GetDim(3);
    const int32_t src_axis_to_dst_batch    = perm[0];
    const int32_t src_axis_to_dst_channles = perm[1];
    const int32_t src_axis_to_dst_height   = perm[2];
    const int32_t src_axis_to_dst_width    = perm[3];
    int64_t src_dims[4]                    = {batch, channels, src_h, src_w};
    int64_t src_stride[4]                  = {int64_t(channels) * src_h * src_w, int64_t(src_h) * src_w, src_w, 1};
    int64_t axis_map[4]                    = {src_axis_to_dst_batch, src_axis_to_dst_channles, src_axis_to_dst_height, src_axis_to_dst_width};
    int64_t dst_dims[4];
    for (int64_t i = 0; i < 4; ++i) {
        dst_dims[i] = src_dims[axis_map[i]];
    }
    int64_t dst_stride[4] = {dst_dims[1] * dst_dims[2] * dst_dims[3], dst_dims[2] * dst_dims[3], dst_dims[3], 1};
    int64_t axis_stride[4];
    for (int64_t i = 0; i < 4; ++i) {
        axis_stride[axis_map[i]] = dst_stride[i];
    }

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t n = 0; n < batch; ++n) {
        for (int32_t c = 0; c < channels; ++c) {
            int64_t batch_in_offset    = n * src_stride[0];
            int64_t batch_out_offset   = n * axis_stride[0];
            int64_t channel_in_offset  = c * src_stride[1];
            int64_t channel_out_offset = c * axis_stride[1];
            for (int64_t h = 0; h < src_h; ++h) {
                int64_t height_in_offset  = h * src_stride[2];
                int64_t height_out_offset = h * axis_stride[2];
                int64_t base_in_offset    = batch_in_offset + channel_in_offset + height_in_offset;
                int64_t base_out_offset   = batch_out_offset + channel_out_offset + height_out_offset;
                for (int64_t w = 0; w < src_w; ++w) {
                    dst[base_out_offset + w * axis_stride[3]] = src[base_in_offset + w];
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
void transpose_ndarray_recursive(
    const single_parallel_loop_config_t &pc,
    const int64_t *src_dims,
    const int64_t *src_stride,
    const int64_t *dst_stride,
    const int32_t *perm,
    const eT *src,
    const uint32_t dim_idx,
    const uint32_t dim_count,
    const int64_t base_in_offset,
    const int64_t base_out_offset,
    eT *dst)
{
    const int64_t length = src_dims[dim_idx];
    if (dim_idx == dim_count - 1) {
        if (dim_idx == pc.depth_of_loop && length > 1) {
            const int64_t len_per_thread = div_up(length, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t start_idx = t * len_per_thread;
                const int64_t end_idx = min<int64_t>(start_idx + len_per_thread, length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    dst[base_out_offset + i * dst_stride[perm[dim_idx]]] = src[base_in_offset + i];
                }
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                dst[base_out_offset + i * dst_stride[perm[dim_idx]]] = src[base_in_offset + i];
            }
        }
    } else {
        if (dim_idx == pc.depth_of_loop && length > 1) {
            const int64_t len_per_thread = div_up(length, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t start_idx = t * len_per_thread;
                const int64_t end_idx = min<int64_t>(start_idx + len_per_thread, length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    transpose_ndarray_recursive<eT>(pc, src_dims, src_stride, dst_stride, perm, src, dim_idx + 1, dim_count, base_in_offset + i * src_stride[dim_idx], base_out_offset + i * dst_stride[perm[dim_idx]], dst);
                }
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                transpose_ndarray_recursive<eT>(pc, src_dims, src_stride, dst_stride, perm, src, dim_idx + 1, dim_count, base_in_offset + i * src_stride[dim_idx], base_out_offset + i * dst_stride[perm[dim_idx]], dst);
            }
        }
    }
}

template <typename eT>
ppl::common::RetCode transpose_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t *perm,
    const eT *src,
    eT *dst)
{
    const uint32_t dim_count = src_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (dim_count <= 1) {
        return ppl::common::RC_SUCCESS;
    }

    if (dim_count == 2) {
        return transpose2d_ndarray(src_shape, src, dst);
    }

    if (dim_count == 3) {
        return transpose3d_ndarray(src_shape, perm, src, dst);
    }

    if (dim_count == 4) {
        return transpose4d_ndarray(src_shape, perm, src, dst);
    }

    auto src_dims = src_shape->GetDims();
    auto dst_dims = dst_shape->GetDims();
    int64_t src_stride[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t dst_stride[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    src_stride[dim_count - 1] = 1;
    dst_stride[dim_count - 1] = 1;
    for (int32_t i = (int32_t)dim_count - 2; i >= 0; i--) {
        src_stride[i] = src_stride[i + 1] * src_dims[i + 1];
        dst_stride[i] = dst_stride[i + 1] * dst_dims[i + 1];
    }

    std::vector<int64_t> loops(src_dims, src_dims + dim_count);
    auto pc = select_single_parallel_loop(loops, ppl::common::ISA_UNKNOWN, sizeof(eT), sizeof(eT), sizeof(eT), 1);

    int32_t revert_perm[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    for (int64_t i = 0; i < dim_count; i++) {
        const int32_t perm_val = perm[i] < 0 ? perm[i] + dim_count : perm[i];
        revert_perm[perm_val] = i;
    }

    transpose_ndarray_recursive(
        pc, src_dims, src_stride, dst_stride, revert_perm, src, 0, dim_count, 0, 0, dst);

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose_ndarray_continous2d(
    const ppl::nn::TensorShape *src_shape,
    const uint32_t axis0,
    const uint32_t axis1,
    const eT *src,
    eT *dst)
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

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t od = 0; od < outer_dims; od++) {
        for (int64_t i = 0; i < dim0; i++) {
            for (int64_t j = 0; j < dim1; j++) {
                const eT *l_src = src + od * dim0 * dim1 * inner_dims + i * dim1 * inner_dims + j * inner_dims;
                eT *l_dst       = dst + od * dim1 * dim0 * inner_dims + j * dim0 * inner_dims + i * inner_dims;
                memcpy(l_dst, l_src, inner_dims * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_
