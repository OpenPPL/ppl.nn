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

#ifndef __ST_PPL_KERNEL_X86_INT64_REDUCE_REDUCE_N16CX_INT64_H_
#define __ST_PPL_KERNEL_X86_INT64_REDUCE_REDUCE_N16CX_INT64_H_

#include "ppl/kernel/x86/int64/reduce/reduce_kernel_int64.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

#define C_BLK() ((int64_t)16)

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_no_reduce_int64(
    const int64_t *src,
    const int64_t width,
    const int64_t remain_c,
    int64_t *dst)
{
    for (int64_t i = 0; i < width; i++) {
        for (int64_t c = 0; c < C_BLK(); c++) {
            dst[i * C_BLK() + c] = reduce_scalar_kernel_int64<_op>(src[i * C_BLK() + c], dst[i * C_BLK() + c]);
        }
    }
}

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_reduce_w_int64(
    const int64_t *src,
    const int64_t width,
    const int64_t remain_c,
    int64_t *dst)
{
    for (int64_t i = 0; i < width; i++) {
        for (int64_t c = 0; c < C_BLK(); c++) {
            dst[c] = reduce_scalar_kernel_int64<_op>(src[i * C_BLK() + c], dst[c]);
        }
    }
}

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_reduce_c_int64(
    const int64_t *src,
    const int64_t width,
    const int64_t remain_c,
    int64_t *dst)
{
    const int64_t c_num = min(remain_c, C_BLK());
    for (int64_t i = 0; i < width; i++) {
        for (int64_t c = 0; c < c_num; c++) {
            dst[i * C_BLK()] = reduce_scalar_kernel_int64<_op>(src[i * C_BLK() + c], dst[i * C_BLK()]);
        }
    }
}

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_reduce_cw_int64(
    const int64_t *src,
    const int64_t width,
    const int64_t remain_c,
    int64_t *dst)
{
    const int64_t init_val = reduce_init_val_int64<_op>();
    int64_t reduce_val     = init_val;

    const int64_t c_num = min(remain_c, C_BLK());
    for (int64_t i = 0; i < width; i++) {
        for (int64_t c = 0; c < c_num; c++) {
            reduce_val = reduce_scalar_kernel_int64<_op>(src[i * C_BLK() + c], reduce_val);
        }
    }
    dst[0] = reduce_scalar_kernel_int64<_op>(reduce_val, dst[0]);
}

template <reduce_op_type_t _op>
void reduce_n16cx_recursive_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int64_t dim_idx,
    const int64_t *inc_src,
    const int64_t *inc_dst,
    const single_parallel_loop_config_t *pc,
    const int64_t c_dim_idx,
    int64_t remain_c,
    int64_t *dst)
{
    if (dim_idx == src_shape->GetDimCount() - 1) { // last dim
        const bool reduce_on_w = src_shape->GetDim(dim_idx) != dst_shape->GetDim(dim_idx);
        const bool reduce_on_c = src_shape->GetDim(c_dim_idx) != dst_shape->GetDim(c_dim_idx);
        const int64_t width    = src_shape->GetDim(dim_idx);
        if (!reduce_on_c && !reduce_on_w) {
            reduce_n16cx_lastdim_no_reduce_int64<_op>(src, width, remain_c, dst);
        } else if (!reduce_on_c && reduce_on_w) {
            reduce_n16cx_lastdim_reduce_w_int64<_op>(src, width, remain_c, dst);
        } else if (reduce_on_c && !reduce_on_w) {
            reduce_n16cx_lastdim_reduce_c_int64<_op>(src, width, remain_c, dst);
        } else { // reduce_on_c && reduce_on_w
            reduce_n16cx_lastdim_reduce_cw_int64<_op>(src, width, remain_c, dst);
        }
    } else {
        const int64_t len = dim_idx == c_dim_idx ? div_up(src_shape->GetDim(dim_idx), C_BLK()) : src_shape->GetDim(dim_idx);
        if (pc->depth_of_loop == dim_idx && pc->num_threads > 1) { // parallel on this dim
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc->num_threads; t++) {
                const int64_t len_per_thread = div_up(len, pc->num_threads);
                const int64_t start_idx      = t * len_per_thread;
                const int64_t end_idx        = min(start_idx + len_per_thread, len);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    if (dim_idx == c_dim_idx) {
                        remain_c = src_shape->GetDim(c_dim_idx) - i * C_BLK();
                    }
                    const int64_t *p_src = src + i * inc_src[dim_idx];
                    int64_t *p_dst       = dst + i * inc_dst[dim_idx];
                    reduce_n16cx_recursive_int64<_op>(src_shape, dst_shape, p_src, dim_idx + 1, inc_src, inc_dst, pc, c_dim_idx, remain_c, p_dst);
                }
            }
        } else {
            for (int64_t i = 0; i < len; i++) {
                if (dim_idx == c_dim_idx) {
                    remain_c = src_shape->GetDim(c_dim_idx) - i * C_BLK();
                }
                const int64_t *p_src = src + i * inc_src[dim_idx];
                int64_t *p_dst       = dst + i * inc_dst[dim_idx];
                reduce_n16cx_recursive_int64<_op>(src_shape, dst_shape, p_src, dim_idx + 1, inc_src, inc_dst, pc, c_dim_idx, remain_c, p_dst);
            }
        }
    }
}

template <reduce_op_type_t _op>
common::RetCode reduce_n16cx_int64(
    const int64_t *src,
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t *axes,
    const int32_t num_axes,
    const int64_t c_dim_idx,
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
    padded_dst_shape.CalcPadding();

    // pre process
    reduce_preprocess_int64<_op>(dst, padded_dst_shape.GetElementsIncludingPadding());

    // prepare incs
    const int64_t dim_count = padded_dst_shape.GetDimCount();
    int64_t inc_src[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_dst[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_src = C_BLK();
    int64_t stride_dst = C_BLK();

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        int64_t src_dim = src_shape->GetDim(i);
        int64_t dst_dim = padded_dst_shape.GetDim(i);
        inc_src[i]      = src_dim == 1 ? 0 : stride_src;
        inc_dst[i]      = dst_dim == 1 ? 0 : stride_dst;

        if (i == c_dim_idx) {
            src_dim = div_up(src_dim, C_BLK());
            dst_dim = div_up(dst_dim, C_BLK());
        }
        stride_src *= src_dim;
        stride_dst *= dst_dim;
    }

    // calc parallel config
    std::vector<int64_t> loop_iter(src_shape->GetDims(), src_shape->GetDims() + dim_count);
    loop_iter[c_dim_idx] = div_up(loop_iter[c_dim_idx], C_BLK());
    std::vector<bool> forbid_mask(dim_count, false);
    for (int64_t i = 0; i < num_axes; i++) { // reduce dims cannot parallel
        forbid_mask[axes[i]] = true;
    }
    forbid_mask[dim_count - 1] = true; // last dim will not use omp because have much overhead when reduce on all before dims, or have error when reduce on last dim

    const bool reduce_on_c = src_shape->GetDim(c_dim_idx) != padded_dst_shape.GetDim(c_dim_idx);
    const bool reduce_on_w = src_shape->GetDim(dim_count - 1) != padded_dst_shape.GetDim(dim_count - 1);
    int64_t load_per_task;
    int64_t store_per_task;
    if (!reduce_on_c) {
        load_per_task  = C_BLK() * 2 * sizeof(int64_t);
        store_per_task = C_BLK() * sizeof(int64_t);
    } else if (!reduce_on_w) {
        load_per_task  = (C_BLK() + 1) * sizeof(int64_t);
        store_per_task = 1 * sizeof(int64_t);
    } else {
        load_per_task  = (C_BLK() + 1) * sizeof(int64_t);
        store_per_task = 0;
    }

    auto pc = select_single_parallel_loop_with_mask(
        loop_iter,
        forbid_mask,
        ppl::common::ISA_UNKNOWN,
        load_per_task,
        store_per_task,
        C_BLK() * sizeof(int64_t),
        1);

    // reduce
    reduce_n16cx_recursive_int64<_op>(src_shape, &padded_dst_shape, src, 0, inc_src, inc_dst, &pc, c_dim_idx, src_shape->GetDim(c_dim_idx), dst);

    // post process
    int64_t reduce_factor = 1;
    for (int64_t i = 0; i < dim_count; i++) {
        reduce_factor *= src_shape->GetDim(i) / padded_dst_shape.GetDim(i);
    }
    reduce_postprocess_int64<_op>(dst, padded_dst_shape.GetElementsIncludingPadding(), reduce_factor);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_INT64_REDUCE_REDUCE_N16CX_INT64_H_
