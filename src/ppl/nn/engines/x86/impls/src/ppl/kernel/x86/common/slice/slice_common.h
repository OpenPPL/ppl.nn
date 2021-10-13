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

#ifndef __ST_PPL_KERNEL_X86_COMMON_SLICE_SLICE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_SLICE_SLICE_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/threading_tools.h"
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode slice_ndarray_recursive(
    const single_parallel_loop_config_t &pc,
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const eT *src,
    const int64_t *starts,
    const int64_t *steps,
    const int64_t *stride_in,
    const int64_t *stride_out,
    const int64_t dim_idx,
    eT *dst)
{
    const int64_t dim_count     = src_shape->GetDimCount();
    const int64_t output_length = dst_shape->GetDim(dim_idx);

    if (dim_idx == dim_count - 1) {
        if (pc.depth_of_loop == dim_idx && output_length > 1) {
            const int64_t len_per_thread = div_up(output_length, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t start_idx = t * len_per_thread;
                const int64_t end_idx = min<int64_t>(start_idx + len_per_thread, output_length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    const int64_t src_i = starts[dim_idx] + i * steps[dim_idx];
                    dst[i]              = src[src_i];
                }
            }
        } else {
            for (int64_t i = 0; i < output_length; i++) {
                const int64_t src_i = starts[dim_idx] + i * steps[dim_idx];
                dst[i]              = src[src_i];
            }
        }
    } else {
        if (pc.depth_of_loop == dim_idx && output_length > 1) {
            const int64_t len_per_thread = div_up(output_length, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t start_idx = t * len_per_thread;
                const int64_t end_idx = min<int64_t>(start_idx + len_per_thread, output_length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    const int64_t src_i = starts[dim_idx] + i * steps[dim_idx];
                    slice_ndarray_recursive<eT>(pc, src_shape, dst_shape, src + src_i * stride_in[dim_idx], starts, steps, stride_in, stride_out, dim_idx + 1, dst + i * stride_out[dim_idx]);
                }
            }
        } else {
            for (int64_t i = 0; i < output_length; i++) {
                const int64_t src_i = starts[dim_idx] + i * steps[dim_idx];
                slice_ndarray_recursive<eT>(pc, src_shape, dst_shape, src + src_i * stride_in[dim_idx], starts, steps, stride_in, stride_out, dim_idx + 1, dst + i * stride_out[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode slice_ndarray_common(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const eT *src,
    const int64_t *starts,
    const int64_t *steps,
    const int64_t *axes,
    const int64_t axes_num,
    eT *dst)
{
    const int64_t dim_count = src_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t real_starts[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t real_steps[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    if (axes_num == dim_count) {
        memcpy(real_starts, starts, axes_num * sizeof(int64_t));
        memcpy(real_steps, steps, axes_num * sizeof(int64_t));
    } else if (axes_num < dim_count) {
        for (int64_t i = 0; i < dim_count; i++) {
            real_starts[i] = 0;
            real_steps[i]  = 1;
        }
        for (int64_t i = 0; i < axes_num; i++) {
            real_starts[axes[i]] = starts[i];
            real_steps[axes[i]]  = steps[i];
        }
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }

    for (int64_t i = 0; i < dim_count; i++) {
        if (real_starts[i] >= src_shape->GetDim(i)) {
            real_starts[i] = src_shape->GetDim(i) - 1;
        }
        if (real_starts[i] < 0) {
            real_starts[i] += src_shape->GetDim(i);
        }
    }

    int64_t stride_in[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    stride_in[dim_count - 1]  = 1;
    stride_out[dim_count - 1] = 1;
    for (int64_t i = dim_count - 2; i >= 0; i--) {
        stride_in[i]  = src_shape->GetDim(i + 1) * stride_in[i + 1];
        stride_out[i] = dst_shape->GetDim(i + 1) * stride_out[i + 1];
    }

    std::vector<int64_t> loops(dst_shape->GetDims(), dst_shape->GetDims() + dim_count);
    auto pc = select_single_parallel_loop(loops, ppl::common::ISA_UNKNOWN, sizeof(eT), sizeof(eT), sizeof(eT), 1);

    return slice_ndarray_recursive<eT>(pc, src_shape, dst_shape, src, real_starts, real_steps, stride_in, stride_out, 0, dst);
}

}}} // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_SLICE_SLICE_COMMON_H_
