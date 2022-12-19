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

#include <vector>
#include <string.h>
#include <algorithm>

#include "ppl/kernel/arm_server/common/memory.h"

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/threading_tools.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

struct continuous_dims {
    int64_t start = 0;
    int64_t end   = 0;
    int64_t idx   = 0;
    bool operator<(const continuous_dims &p) const
    {
        return this->start < p.start;
    }
};

template <typename eT>
static ppl::common::RetCode transpose2d_ndarray_common(
    const std::vector<int64_t> &src_dims,
    const eT *src,
    eT *dst)
{
    const int64_t src_h = src_dims[0];
    const int64_t src_w = src_dims[1];

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < src_h; ++i) {
        int64_t j = 0;
        for (; j <= src_w - 8; j += 8) {
            dst[(j + 0) * src_h + i] = src[i * src_w + j + 0];
            dst[(j + 1) * src_h + i] = src[i * src_w + j + 1];
            dst[(j + 2) * src_h + i] = src[i * src_w + j + 2];
            dst[(j + 3) * src_h + i] = src[i * src_w + j + 3];
            dst[(j + 4) * src_h + i] = src[i * src_w + j + 4];
            dst[(j + 5) * src_h + i] = src[i * src_w + j + 5];
            dst[(j + 6) * src_h + i] = src[i * src_w + j + 6];
            dst[(j + 7) * src_h + i] = src[i * src_w + j + 7];
        }
        for (; j < src_w; ++j) {
            dst[j * src_h + i] = src[i * src_w + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose3d_ndarray_common(
    const std::vector<int64_t> &src_dims,
    const std::vector<int64_t> &perm,
    const eT *src,
    eT *dst)
{
    const int64_t channels                 = src_dims[0];
    const int64_t src_h                    = src_dims[1];
    const int64_t src_w                    = src_dims[2];
    const int64_t src_axis_to_dst_channels = perm[0];
    const int64_t src_axis_to_dst_height   = perm[1];
    const int64_t src_axis_to_dst_width    = perm[2];
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
    for (int64_t c = 0; c < channels; ++c) {
        int64_t channel_in_offset  = c * src_stride[0];
        int64_t channel_out_offset = c * axis_stride[0];
        for (int64_t h = 0; h < src_h; ++h) {
            int64_t height_in_offset  = h * src_stride[1];
            int64_t height_out_offset = h * axis_stride[1];
            int64_t base_in_offset    = channel_in_offset + height_in_offset;
            int64_t base_out_offset   = channel_out_offset + height_out_offset;
            int64_t w                 = 0;
            for (; w <= src_w - 8; w += 8) {
                dst[base_out_offset + (w + 0) * axis_stride[2]] = src[base_in_offset + w + 0];
                dst[base_out_offset + (w + 1) * axis_stride[2]] = src[base_in_offset + w + 1];
                dst[base_out_offset + (w + 2) * axis_stride[2]] = src[base_in_offset + w + 2];
                dst[base_out_offset + (w + 3) * axis_stride[2]] = src[base_in_offset + w + 3];
                dst[base_out_offset + (w + 4) * axis_stride[2]] = src[base_in_offset + w + 4];
                dst[base_out_offset + (w + 5) * axis_stride[2]] = src[base_in_offset + w + 5];
                dst[base_out_offset + (w + 6) * axis_stride[2]] = src[base_in_offset + w + 6];
                dst[base_out_offset + (w + 7) * axis_stride[2]] = src[base_in_offset + w + 7];
            }
            for (; w < src_w; ++w) {
                dst[base_out_offset + w * axis_stride[2]] = src[base_in_offset + w];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose4d_ndarray_common(
    const std::vector<int64_t> &src_dims,
    const std::vector<int64_t> &perm,
    const eT *src,
    eT *dst)
{
    const int64_t batch                    = src_dims[0];
    const int64_t channels                 = src_dims[1];
    const int64_t src_h                    = src_dims[2];
    const int64_t src_w                    = src_dims[3];
    const int64_t src_axis_to_dst_batch    = perm[0];
    const int64_t src_axis_to_dst_channles = perm[1];
    const int64_t src_axis_to_dst_height   = perm[2];
    const int64_t src_axis_to_dst_width    = perm[3];
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
        for (int64_t c = 0; c < channels; ++c) {
            int64_t batch_in_offset    = n * src_stride[0];
            int64_t batch_out_offset   = n * axis_stride[0];
            int64_t channel_in_offset  = c * src_stride[1];
            int64_t channel_out_offset = c * axis_stride[1];
            for (int64_t h = 0; h < src_h; ++h) {
                int64_t height_in_offset  = h * src_stride[2];
                int64_t height_out_offset = h * axis_stride[2];
                int64_t base_in_offset    = batch_in_offset + channel_in_offset + height_in_offset;
                int64_t base_out_offset   = batch_out_offset + channel_out_offset + height_out_offset;
                int64_t w                 = 0;
                for (; w <= src_w - 8; w += 8) {
                    dst[base_out_offset + (w + 0) * axis_stride[3]] = src[base_in_offset + w + 0];
                    dst[base_out_offset + (w + 1) * axis_stride[3]] = src[base_in_offset + w + 1];
                    dst[base_out_offset + (w + 2) * axis_stride[3]] = src[base_in_offset + w + 2];
                    dst[base_out_offset + (w + 3) * axis_stride[3]] = src[base_in_offset + w + 3];
                    dst[base_out_offset + (w + 4) * axis_stride[3]] = src[base_in_offset + w + 4];
                    dst[base_out_offset + (w + 5) * axis_stride[3]] = src[base_in_offset + w + 5];
                    dst[base_out_offset + (w + 6) * axis_stride[3]] = src[base_in_offset + w + 6];
                    dst[base_out_offset + (w + 7) * axis_stride[3]] = src[base_in_offset + w + 7];
                }
                for (; w < src_w; ++w) {
                    dst[base_out_offset + w * axis_stride[3]] = src[base_in_offset + w];
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose_ndarray_continous2d_common(
    const std::vector<int64_t> &src_dims,
    const int64_t axis0,
    const int64_t axis1,
    const eT *src,
    eT *dst)
{
    const int64_t dim_count = src_dims.size();
    const int64_t dim0      = src_dims[axis0];
    const int64_t dim1      = src_dims[axis1];

    int64_t outer_dims = 1;
    for (int64_t i = 0; i < axis0; i++) {
        outer_dims *= src_dims[i];
    }

    int64_t inner_dims = 1;
    for (int64_t i = axis1 + 1; i < dim_count; i++) {
        inner_dims *= src_dims[i];
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t od = 0; od < outer_dims; od++) {
        for (int64_t i = 0; i < dim0; i++) {
            for (int64_t j = 0; j < dim1; j++) {
                const eT *l_src = src + od * dim0 * dim1 * inner_dims + i * dim1 * inner_dims + j * inner_dims;
                eT *l_dst       = dst + od * dim0 * dim1 * inner_dims + j * dim0 * inner_dims + i * inner_dims;
                memcpy(l_dst, l_src, inner_dims * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
static ppl::common::RetCode transpose_ndarray_recursive_common(
    const single_parallel_loop_config_t &pc,
    const std::vector<int64_t> &src_dims,
    const std::vector<int64_t> &src_stride,
    const std::vector<int64_t> &dst_stride,
    const std::vector<int64_t> &revert_perm,
    const eT *src,
    const int64_t dim_idx,
    const int64_t dim_count,
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
                const int64_t end_idx   = min<int64_t>(start_idx + len_per_thread, length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    dst[base_out_offset + i * dst_stride[revert_perm[dim_idx]]] = src[base_in_offset + i];
                }
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                dst[base_out_offset + i * dst_stride[revert_perm[dim_idx]]] = src[base_in_offset + i];
            }
        }
    } else {
        if (dim_idx == pc.depth_of_loop && length > 1) {
            const int64_t len_per_thread = div_up(length, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t start_idx = t * len_per_thread;
                const int64_t end_idx   = min<int64_t>(start_idx + len_per_thread, length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    transpose_ndarray_recursive_common<eT>(pc, src_dims, src_stride, dst_stride, revert_perm, src, dim_idx + 1, dim_count, base_in_offset + i * src_stride[dim_idx], base_out_offset + i * dst_stride[revert_perm[dim_idx]], dst);
                }
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                transpose_ndarray_recursive_common<eT>(pc, src_dims, src_stride, dst_stride, revert_perm, src, dim_idx + 1, dim_count, base_in_offset + i * src_stride[dim_idx], base_out_offset + i * dst_stride[revert_perm[dim_idx]], dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline void merge_transpose_dims(
    const int64_t *src_dims,
    const int64_t dim_count,
    const int64_t *perm,
    std::vector<int64_t> &merged_src_dims,
    std::vector<int64_t> &merged_perm)
{
    std::vector<continuous_dims> to_merge_dims;
    continuous_dims dims;
    dims.start = dims.end = perm[0];
    for (int64_t i = 1; i < dim_count; i++) {
        if (perm[i] - perm[i - 1] == 1) {
            dims.end = perm[i];
        } else {
            dims.idx = to_merge_dims.size();
            to_merge_dims.push_back(dims);
            dims.start = dims.end = perm[i];
        }
    }
    dims.idx = to_merge_dims.size();
    to_merge_dims.push_back(dims);

    std::sort(to_merge_dims.begin(), to_merge_dims.end());

    const int64_t merged_dim_count = to_merge_dims.size();
    merged_src_dims.resize(merged_dim_count);
    merged_perm.resize(merged_dim_count);

    for (int64_t i = 0; i < merged_dim_count; i++) {
        continuous_dims merge_dims  = to_merge_dims[i];
        merged_perm[merge_dims.idx] = i;
        int64_t dim_value           = 1;
        for (int64_t j = merge_dims.start; j <= merge_dims.end; j++) {
            dim_value *= src_dims[j];
        }
        merged_src_dims[i] = dim_value;
    }
}

template <typename eT>
static ppl::common::RetCode transpose_ndarray_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const int64_t *perm,
    const eT *src,
    eT *dst)
{
    const int64_t dim_count = src_shape->GetDimCount();
    const int64_t *src_dims = src_shape->GetDims();

    std::vector<int64_t> merged_src_dims;
    std::vector<int64_t> merged_perm;
    merge_transpose_dims(src_dims, dim_count, perm, merged_src_dims, merged_perm);

    const int64_t merged_dim_count = merged_src_dims.size();
    if (merged_dim_count >= 3) {
        std::vector<int64_t> transposed_dim;
        transposed_dim.reserve(merged_dim_count);
        for (int64_t i = 0; i < merged_dim_count; i++) {
            if (merged_perm[i] != i) {
                transposed_dim.push_back(i);
            }
        }
        if (transposed_dim.size() == 2 && transposed_dim[0] + 1 == transposed_dim[1] && transposed_dim[1] + 1 < merged_dim_count) {
            return transpose_ndarray_continous2d_common<eT>(merged_src_dims, transposed_dim[0], transposed_dim[1], src, dst);
        }
    }

    if (merged_dim_count == 1) {
        return memory_copy(src, src_shape->CalcBytesIncludingPadding(), dst);
    }
    if (merged_dim_count == 2) {
        return transpose2d_ndarray_common<eT>(merged_src_dims, src, dst);
    }
    if (merged_dim_count == 3) {
        return transpose3d_ndarray_common<eT>(merged_src_dims, merged_perm, src, dst);
    }
    if (merged_dim_count == 4) {
        return transpose4d_ndarray_common<eT>(merged_src_dims, merged_perm, src, dst);
    }

    std::vector<int64_t> merged_dst_dims(merged_dim_count);
    for (int64_t i = 0; i < merged_dim_count; i++) {
        merged_dst_dims[i] = merged_src_dims[merged_perm[i]];
    }

    std::vector<int64_t> src_stride(merged_dim_count, 1);
    std::vector<int64_t> dst_stride(merged_dim_count, 1);
    for (int64_t i = merged_dim_count - 2; i >= 0; i--) {
        src_stride[i] = src_stride[i + 1] * merged_src_dims[i + 1];
        dst_stride[i] = dst_stride[i + 1] * merged_dst_dims[i + 1];
    }

    std::vector<int64_t> revert_perm(merged_dim_count);
    for (int64_t i = 0; i < merged_dim_count; i++) {
        revert_perm[merged_perm[i]] = i;
    }

    const float omp_div_task_time_ratio       = 20.0f; // assume omp create thread may be 20x slower than one element process
    single_parallel_loop_config_t loop_config = select_single_parallel_loop(merged_src_dims, omp_div_task_time_ratio);

    return transpose_ndarray_recursive_common<eT>(
        loop_config,
        merged_src_dims,
        src_stride,
        dst_stride,
        revert_perm,
        src,
        0,
        merged_dim_count,
        0,
        0,
        dst);
}

ppl::common::RetCode transpose(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const int64_t *perm,
    const void *src,
    void *dst)
{
    const auto data_type   = src_shape->GetDataType();
    const auto data_format = src_shape->GetDataFormat();
    if (data_format != ppl::common::DATAFORMAT_NDARRAY) {
        return ppl::common::RC_UNSUPPORTED;
    }

    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return transpose_ndarray_common<uint8_t>(src_shape, dst_shape, perm, (const uint8_t *)src, (uint8_t *)dst);
        case 2: return transpose_ndarray_common<uint16_t>(src_shape, dst_shape, perm, (const uint16_t *)src, (uint16_t *)dst);
        case 4: return transpose_ndarray_common<uint32_t>(src_shape, dst_shape, perm, (const uint32_t *)src, (uint32_t *)dst);
        case 8: return transpose_ndarray_common<uint64_t>(src_shape, dst_shape, perm, (const uint64_t *)src, (uint64_t *)dst);
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
