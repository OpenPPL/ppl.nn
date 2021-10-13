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

#ifndef __ST_PPL_KERNEL_X86_INT64_REDUCE_REDUCE_NDARRAY_INT64_H_
#define __ST_PPL_KERNEL_X86_INT64_REDUCE_REDUCE_NDARRAY_INT64_H_

#include "ppl/kernel/x86/int64/reduce/reduce_kernel_int64.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

template <reduce_op_type_t _op>
static void reduce_ndarray_lastdim_no_reduce_int64(
    const int64_t *src,
    const int64_t width,
    int64_t *dst)
{
    for (int64_t i = 0; i < width; i++) {
        dst[i] = reduce_scalar_kernel_int64<_op>(src[i], dst[i]);
    }
}

template <reduce_op_type_t _op>
static void reduce_ndarray_lastdim_reduce_int64(
    const int64_t *src,
    const int64_t width,
    int64_t *dst)
{
    const int64_t init_val = reduce_init_val_int64<_op>();
    int64_t reduce_val     = init_val;

    for (int64_t i = 0; i < width; i++) {
        reduce_val = reduce_scalar_kernel_int64<_op>(src[i], reduce_val);
    }
    dst[0] = reduce_scalar_kernel_int64<_op>(reduce_val, dst[0]);
}

template <reduce_op_type_t _op>
static void reduce_ndarray_recursive_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int64_t dim_idx,
    const int64_t *inc_src,
    const int64_t *inc_dst,
    const single_parallel_loop_config_t *pc,
    int64_t *dst)
{
    const int64_t len = src_shape->GetDim(dim_idx);

    if (dim_idx == src_shape->GetDimCount() - 1) { // last dim
        if (len == dst_shape->GetDim(dim_idx)) {
            reduce_ndarray_lastdim_no_reduce_int64<_op>(src, len, dst);
        } else { // reduce on last dim_idx
            reduce_ndarray_lastdim_reduce_int64<_op>(src, len, dst);
        }
    } else {
        if (pc->depth_of_loop == dim_idx && pc->num_threads > 1) { // parallel on this dim
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc->num_threads; t++) {
                const int64_t len_per_thread = div_up(len, pc->num_threads);
                const int64_t start_idx      = len_per_thread * t;
                const int64_t end_idx        = min(start_idx + len_per_thread, len);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    const int64_t *p_src = src + i * inc_src[dim_idx];
                    int64_t *p_dst       = dst + i * inc_dst[dim_idx];
                    reduce_ndarray_recursive_int64<_op>(src_shape, dst_shape, p_src, dim_idx + 1, inc_src, inc_dst, pc, p_dst);
                }
            }
        } else {
            for (int64_t i = 0; i < len; i++) {
                const int64_t *p_src = src + i * inc_src[dim_idx];
                int64_t *p_dst       = dst + i * inc_dst[dim_idx];
                reduce_ndarray_recursive_int64<_op>(src_shape, dst_shape, p_src, dim_idx + 1, inc_src, inc_dst, pc, p_dst);
            }
        }
    }
}

template <reduce_op_type_t _op>
ppl::common::RetCode reduce_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    if (src_shape->GetDimCount() > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    // pad 1 to dst shape to keepdims
    ppl::nn::TensorShape padded_dst_shape = *src_shape;
    for (int64_t i = 0; i < num_axes; i++) {
        padded_dst_shape.SetDim(axes[i], 1);
    }

    // pre process
    reduce_preprocess_int64<_op>(dst, padded_dst_shape.GetElementsIncludingPadding());

    // prepare incs
    const int64_t dim_count = padded_dst_shape.GetDimCount();
    int64_t inc_src[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_dst[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_src = 1;
    int64_t stride_dst = 1;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc_src[i] = src_shape->GetDim(i) == 1 ? 0 : stride_src;
        inc_dst[i] = padded_dst_shape.GetDim(i) == 1 ? 0 : stride_dst;

        stride_src *= src_shape->GetDim(i);
        stride_dst *= padded_dst_shape.GetDim(i);
    }

    // calc parallel config
    std::vector<int64_t> loop_iter(src_shape->GetDims(), src_shape->GetDims() + dim_count);
    std::vector<bool> forbid_mask(dim_count, false);
    for (int64_t i = 0; i < num_axes; i++) { // reduce dims cannot parallel
        forbid_mask[axes[i]] = true;
    }
    forbid_mask[dim_count - 1]    = true; // last dim will not use omp because have much overhead when reduce on all before dims, or have error when reduce on last dim
    const bool reduce_on_last_dim = src_shape->GetDim(dim_count - 1) != padded_dst_shape.GetDim(dim_count - 1);

    auto pc = select_single_parallel_loop_with_mask(
        loop_iter,
        forbid_mask,
        ppl::common::ISA_UNKNOWN,
        reduce_on_last_dim ? sizeof(int64_t) : 2 * sizeof(int64_t),
        reduce_on_last_dim ? 0 : sizeof(int64_t),
        sizeof(int64_t),
        1);

    // reduce
    reduce_ndarray_recursive_int64<_op>(src_shape, &padded_dst_shape, src, 0, inc_src, inc_dst, &pc, dst);

    // post process
    int64_t reduce_factor = 1;
    for (int64_t i = 0; i < dim_count; i++) {
        reduce_factor *= src_shape->GetDim(i) / padded_dst_shape.GetDim(i);
    }
    reduce_postprocess_int64<_op>(dst, padded_dst_shape.GetElementsIncludingPadding(), reduce_factor);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_INT64_REDUCE_REDUCE_NDARRAY_INT64_H_
