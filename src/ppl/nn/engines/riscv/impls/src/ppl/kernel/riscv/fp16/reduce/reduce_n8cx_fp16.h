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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_REDUCE_REDUCE_N8CX_FP16_H_
#define __ST_PPL_KERNEL_RISCV_FP16_REDUCE_REDUCE_N8CX_FP16_H_

#include "ppl/kernel/riscv/fp16/reduce/reduce_kernel_fp16.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

template <reduce_op_type_t op>
void reduce_n8cx_lastdim_no_reduce_fp16(
    const __fp16* src,
    __fp16* dst,

    const int64_t dim_length,
    const int64_t remain_c)
{
    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    int64_t i = 0;
    for (; i + unroll_len < dim_length * 8; i += unroll_len) {
        vsev_float16xm1(dst + i + 0 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 0 * 8, vl), vlev_float16xm1(dst + i + 0 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 1 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 1 * 8, vl), vlev_float16xm1(dst + i + 1 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 2 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 2 * 8, vl), vlev_float16xm1(dst + i + 2 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 3 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 3 * 8, vl), vlev_float16xm1(dst + i + 3 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 4 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 4 * 8, vl), vlev_float16xm1(dst + i + 4 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 5 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 5 * 8, vl), vlev_float16xm1(dst + i + 5 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 6 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 6 * 8, vl), vlev_float16xm1(dst + i + 6 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 7 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 7 * 8, vl), vlev_float16xm1(dst + i + 7 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 8 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 8 * 8, vl), vlev_float16xm1(dst + i + 8 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 9 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 9 * 8, vl), vlev_float16xm1(dst + i + 9 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 10 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 10 * 8, vl), vlev_float16xm1(dst + i + 10 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 11 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 11 * 8, vl), vlev_float16xm1(dst + i + 11 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 12 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 12 * 8, vl), vlev_float16xm1(dst + i + 12 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 13 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 13 * 8, vl), vlev_float16xm1(dst + i + 13 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 14 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 14 * 8, vl), vlev_float16xm1(dst + i + 14 * 8, vl)), vl);
        vsev_float16xm1(dst + i + 15 * 8, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i + 15 * 8, vl), vlev_float16xm1(dst + i + 15 * 8, vl)), vl);
    }
    for (; i < dim_length * 8; i += 8) {
        vsev_float16xm1(dst + i, reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i, vl), vlev_float16xm1(dst + i, vl)), vl);
    }
}

template <reduce_op_type_t op>
void reduce_n8cx_lastdim_reduce_w_fp16(
    const __fp16* src,
    __fp16* dst,

    const int64_t dim_length,
    const int64_t remain_c)
{
    const int64_t parall_d   = 1;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    float16xm1_t v_reduce_val = vlev_float16xm1(dst, vl);

    int64_t i = 0;
    for (; i < dim_length * 8; i += unroll_len) {
        v_reduce_val = reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i, vl), v_reduce_val);
    }
    vsev_float16xm1(dst, v_reduce_val, vl);
}

template <reduce_op_type_t op>
void reduce_n8cx_lastdim_reduce_c_fp16(
    const __fp16* src,
    __fp16* dst,

    const int64_t dim_length,
    const int64_t remain_c)
{
    const auto vl = vsetvli(8, RVV_E16, RVV_M1);
    if (remain_c >= 8) {
        int64_t i = 0;
        for (; i < dim_length * 8; i += 8) {
            __fp16 reduce_val = reduce_vector_all_lanes_kernel_fp16<op>(vlev_float16xm1(src + i, vl));
            dst[i]            = reduce_scalar_kernel_fp16<op>(dst[i], reduce_val);
        }
    } else { // if remain_c is aligned to C_BLK(), this branch is useless -- make sure 'src_shape[1]' is aligned
    }
}

template <reduce_op_type_t op>
void reduce_n8cx_lastdim_reduce_cw_fp16(
    const __fp16* src,
    __fp16* dst,

    const int64_t dim_length,
    const int64_t remain_c)
{
    const auto vl = vsetvli(8, RVV_E16, RVV_M1);

    if (remain_c >= 8) {
        float16xm1_t v_reduce_val = vfmvvf_float16xm1(reduce_init_val_fp16<op>(), vl);

        int64_t i = 0;
        for (; i < dim_length * 8; i += 8) {
            v_reduce_val = reduce_vector_kernel_fp16<op>(vlev_float16xm1(src + i, vl), v_reduce_val);
        }
        __fp16 reduce_val = reduce_vector_all_lanes_kernel_fp16<op>(v_reduce_val);
        dst[0]            = reduce_scalar_kernel_fp16<op>(reduce_val, dst[0]);
    } else { // if remain_c is aligned to C_BLK(), this branch is useless -- make sure 'src_shape[1]' is aligned
    }
}

template <reduce_op_type_t op>
void reduce_n8cx_recursive_fp16(
    const __fp16* src,
    __fp16* dst,

    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const int64_t dim_idx,
    const int64_t* inc_src,
    const int64_t* inc_dst,
    const int64_t c_dim_idx,
    int64_t remain_c)
{
    if (dim_idx == src_shape->GetDimCount() - 1) {
        const bool reduce_on_w   = src_shape->GetDim(dim_idx) != dst_shape->GetDim(dim_idx);
        const bool reduce_on_c   = src_shape->GetDim(c_dim_idx) != dst_shape->GetDim(c_dim_idx);
        const int64_t dim_length = src_shape->GetDim(dim_idx);
        if (!reduce_on_c && !reduce_on_w) {
            reduce_n8cx_lastdim_no_reduce_fp16<op>(src, dst, dim_length, remain_c);
        } else if (!reduce_on_c && reduce_on_w) {
            reduce_n8cx_lastdim_reduce_w_fp16<op>(src, dst, dim_length, remain_c);
        } else if (reduce_on_c && !reduce_on_w) {
            reduce_n8cx_lastdim_reduce_c_fp16<op>(src, dst, dim_length, remain_c);
        } else {
            reduce_n8cx_lastdim_reduce_cw_fp16<op>(src, dst, dim_length, remain_c);
        }
    } else {
        const int64_t len = dim_idx == c_dim_idx ? div_up(src_shape->GetDim(dim_idx), 8) : src_shape->GetDim(dim_idx);
        for (int64_t i = 0; i < len; i++) {
            if (dim_idx == c_dim_idx) {
                remain_c = src_shape->GetDim(c_dim_idx) - i * 8;
            }
            reduce_n8cx_recursive_fp16<op>(src + i * inc_src[dim_idx], dst + i * inc_dst[dim_idx], src_shape, dst_shape, dim_idx + 1, inc_src, inc_dst, c_dim_idx, remain_c);
        }
    }
}

template <reduce_op_type_t op>
ppl::common::RetCode reduce_n8cx_fp16(
    const __fp16* src,
    __fp16* dst,

    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const int32_t* axes,
    const int32_t num_axes,
    const int64_t c_dim_idx)
{
    if (src_shape->GetDimCount() > PPL_RISCV_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    ppl::common::TensorShape& padded_dst_shape = *(new ppl::common::TensorShape(*src_shape));
    for (int64_t i = 0; i < num_axes; i++) {
        padded_dst_shape.SetDim(axes[i], 1);
    }
    padded_dst_shape.CalcPadding();

    reduce_preprocess_fp16<op>(dst, padded_dst_shape.CalcElementsIncludingPadding());

    int64_t dim_count                            = padded_dst_shape.GetDimCount();
    int64_t inc_src[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_dst[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_src                           = 8;
    int64_t stride_dst                           = 8;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        int64_t src_dim = src_shape->GetDim(i);
        int64_t dst_dim = padded_dst_shape.GetDim(i);
        inc_src[i]      = src_dim == 1 ? 0 : stride_src;
        inc_dst[i]      = dst_dim == 1 ? 0 : stride_dst;

        if (i == c_dim_idx) {
            src_dim = div_up(src_dim, 8);
            dst_dim = div_up(dst_dim, 8);
        }
        stride_src *= src_dim;
        stride_dst *= dst_dim;
    }

    reduce_n8cx_recursive_fp16<op>(src, dst, src_shape, &padded_dst_shape, 0, inc_src, inc_dst, c_dim_idx, src_shape->GetDim(c_dim_idx));

    int64_t reduce_factor = 1;
    for (int64_t i = 0; i < dim_count; i++) {
        reduce_factor *= src_shape->GetDim(i) / padded_dst_shape.GetDim(i);
    }

    reduce_postprocess_fp16<op>(dst, padded_dst_shape.CalcElementsIncludingPadding(), reduce_factor);

    delete &padded_dst_shape;

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_REDUCE_REDUCE_N8CX_FP16_H_
