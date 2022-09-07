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

#ifndef __ST_PPL_KERNEL_X86_COMMON_CONCAT_CONCAT_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_CONCAT_CONCAT_COMMON_H_

#include <vector>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

struct concat_parallel_info {
    concat_parallel_info() {}
    concat_parallel_info(
        const int32_t blk_idx,
        const int32_t start,
        const int32_t end)
    {
        this->blk_idx = blk_idx;
        this->start   = start;
        this->end     = end;
    }

    int32_t blk_idx;
    int32_t start;
    int32_t end;
};

template <typename eT>
ppl::common::RetCode concat_ndarray(
    const ppl::nn::TensorShape **src_shape_list,
    const eT **src_list,
    const int32_t num_src,
    const int32_t axis,
    eT *dst)
{
    const int32_t ndims      = int32_t(src_shape_list[0]->GetDimCount());
    const int32_t fixed_axis = axis < 0 ? ndims + axis : axis;

    int64_t outer_dim = 1;
    int64_t inner_dim = 1;
    for (int32_t i = 0; i < fixed_axis; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int32_t i = fixed_axis + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0]       = 0;
    for (int32_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(fixed_axis);
    }
    const int64_t dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(fixed_axis);

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (dst_concat_dim * inner_dim * sizeof(eT) < 16 && num_src >= num_threads) { // when has small inner dims(usually occurred when fixed_axis == ndims - 1), use scalar code to replace memcpy & change index calculating method
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t n = 0; n < num_src; n++) {
            eT *p_dst                    = dst + dst_offset[n] * inner_dim;
            const eT *p_src              = src_list[n];
            const int64_t copy_elements = src_shape_list[n]->GetDim(fixed_axis) * inner_dim;
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t j = 0; j < copy_elements; j++) {
                    p_dst[j] = p_src[j];
                }
                p_dst += dst_concat_dim * inner_dim;
                p_src += copy_elements;
            }
        }
    } else if (num_src * outer_dim >= num_threads) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t i = 0; i < outer_dim; i++) {
            for (int64_t n = 0; n < num_src; n++) {
                memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                       src_list[n] + i * src_shape_list[n]->GetDim(fixed_axis) * inner_dim,
                       src_shape_list[n]->GetDim(fixed_axis) * inner_dim * sizeof(eT));
            }
        }
    } else { // parallel divide along concat dim, may across different src
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), dst_concat_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t concat_dim_per_thread = div_up(dst_concat_dim, num_threads);
            const int64_t start_concat_dim      = concat_dim_per_thread * thread_id;
            const int64_t end_concat_dim        = min(concat_dim_per_thread * (thread_id + 1), dst_concat_dim);

            if (start_concat_dim < end_concat_dim) {
                int64_t start_blk_idx = num_src - 1;
                int64_t end_blk_idx   = num_src - 1;
                for (int64_t i = 0; i < num_src - 1; i++) {
                    if (start_concat_dim >= dst_offset[i] && start_concat_dim < dst_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_concat_dim >= dst_offset[i] && end_concat_dim < dst_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_concat_dim - dst_offset[start_blk_idx];
                int64_t end_axis_idx   = end_concat_dim - dst_offset[end_blk_idx];

                std::vector<concat_parallel_info> infos;
                if (start_blk_idx == end_blk_idx) { // copy from single src
                    infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                } else { // start_blk_idx < end_blk_idx, copy from multiple src
                    infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, src_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                    for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                        infos.emplace_back(concat_parallel_info(i, 0, src_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                    }
                    infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                }

                for (int64_t i = 0; i < outer_dim; i++) {
                    for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                        const concat_parallel_info &info = infos[j];

                        const eT *p_src = src_list[info.blk_idx] + (i * src_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dim;
                        eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                        const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                        memcpy(p_dst, p_src, size);
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template<typename eT, int32_t channels>
inline void concat_n16cx_interleave_kernel(
    const eT *src[16],
    const int64_t &inner_start,
    const int64_t &inner_end,
    eT *dst)
{
    const int64_t c_blk = 16;
    for (int64_t l = inner_start; l < inner_end; ++l) {
        if (channels > 0 ) dst[l * c_blk + 0 ] = src[0 ][l * c_blk];
        if (channels > 1 ) dst[l * c_blk + 1 ] = src[1 ][l * c_blk];
        if (channels > 2 ) dst[l * c_blk + 2 ] = src[2 ][l * c_blk];
        if (channels > 3 ) dst[l * c_blk + 3 ] = src[3 ][l * c_blk];

        if (channels > 4 ) dst[l * c_blk + 4 ] = src[4 ][l * c_blk];
        if (channels > 5 ) dst[l * c_blk + 5 ] = src[5 ][l * c_blk];
        if (channels > 6 ) dst[l * c_blk + 6 ] = src[6 ][l * c_blk];
        if (channels > 7 ) dst[l * c_blk + 7 ] = src[7 ][l * c_blk];

        if (channels > 8 ) dst[l * c_blk + 8 ] = src[8 ][l * c_blk];
        if (channels > 9 ) dst[l * c_blk + 9 ] = src[9 ][l * c_blk];
        if (channels > 10) dst[l * c_blk + 10] = src[10][l * c_blk];
        if (channels > 11) dst[l * c_blk + 11] = src[11][l * c_blk];

        if (channels > 12) dst[l * c_blk + 12] = src[12][l * c_blk];
        if (channels > 13) dst[l * c_blk + 13] = src[13][l * c_blk];
        if (channels > 14) dst[l * c_blk + 14] = src[14][l * c_blk];
        if (channels > 15) dst[l * c_blk + 15] = src[15][l * c_blk];
    }
}

template <typename eT>
ppl::common::RetCode concat_n16cx_interleave_channels(
    const ppl::nn::TensorShape **src_shape_list,
    const eT **src_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx,
    eT *dst)
{
    const int32_t ndims = int32_t(src_shape_list[0]->GetDimCount());
    int64_t outer_dim   = 1;
    int64_t inner_dim   = 1;
    for (int32_t i = 0; i < c_dim_idx; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int32_t i = c_dim_idx + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src + 1);
    dst_offset[0] = 0;
    for (int32_t i = 1; i <= num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t c_blk              = 16;
    const int64_t dst_channels       = dst_offset[num_src];
    const int64_t padded_oc          = round_up(dst_channels, c_blk);
    const int64_t INNER_PALL_BLK_LEN = inner_dim / (PPL_OMP_NUM_THREADS() / (outer_dim * padded_oc / c_blk) + 1) + 1;

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
    for (int64_t i = 0; i < outer_dim; i++) {
        for (int64_t oc = 0; oc < dst_channels; oc += c_blk) {
            for (int64_t inner = 0; inner < inner_dim; inner += INNER_PALL_BLK_LEN) {
                const int64_t inner_start = inner;
                const int64_t inner_end   = min(inner + INNER_PALL_BLK_LEN, inner_dim);
                const int64_t oc_len_eff  = min(dst_channels - oc, c_blk);
                eT *base_dst              = dst + i * padded_oc * inner_dim + oc * inner_dim;
                const eT *base_src[16]    = {0};
                
                int32_t ic_num   = 0;
                int32_t pre_id   = -1;
                int32_t first_ic = -1;
                for (int64_t j = 0; j < oc_len_eff; j++) {
                    int64_t ic_id = 0;
                    int64_t ic    = 0;
                    for (int64_t idx = 0; idx < num_src; idx++) {
                        if (oc + j < dst_offset[idx + 1]) {
                            ic_id = idx;
                            ic    = oc + j - dst_offset[idx];
                            if (ic_id != pre_id) {
                                ic_num++;
                                pre_id = ic_id;
                            }
                            if (j == 0) {
                                first_ic = ic;
                            }
                            break;
                        }
                    }
                    const int32_t src_channels   = src_shape_list[ic_id]->GetDim(c_dim_idx);
                    const int64_t padded_ic      = round_up(src_channels, c_blk);
                    const int64_t padded_ic_down = round(ic, c_blk);
                    base_src[j]                  = src_list[ic_id] + i * padded_ic * inner_dim + padded_ic_down * inner_dim + ic % c_blk;
                }

                if (base_src[0] + 15 == base_src[15]) {
                    memcpy(base_dst + inner_start * c_blk, base_src[0] + inner_start * c_blk, (inner_end - inner_start) * c_blk * sizeof(eT));
                    continue;
                }

                if (base_src[15] != 0 && ic_num == 1) {
                    const int32_t c_offset = c_blk - (first_ic % c_blk);
                    for (int64_t l = inner_start; l < inner_end; l++) {
                        memcpy(base_dst + l * c_blk,            base_src[0]        + l * c_blk, c_offset * sizeof(eT));
                        memcpy(base_dst + l * c_blk + c_offset, base_src[c_offset] + l * c_blk, (c_blk - c_offset) * sizeof(eT));
                    }
                    continue;
                }

                if      (oc_len_eff == 16) concat_n16cx_interleave_kernel<eT, 16>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 15) concat_n16cx_interleave_kernel<eT, 15>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 14) concat_n16cx_interleave_kernel<eT, 14>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 13) concat_n16cx_interleave_kernel<eT, 13>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 12) concat_n16cx_interleave_kernel<eT, 12>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 11) concat_n16cx_interleave_kernel<eT, 11>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 10) concat_n16cx_interleave_kernel<eT, 10>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 9 ) concat_n16cx_interleave_kernel<eT, 9 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 8 ) concat_n16cx_interleave_kernel<eT, 8 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 7 ) concat_n16cx_interleave_kernel<eT, 7 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 6 ) concat_n16cx_interleave_kernel<eT, 6 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 5 ) concat_n16cx_interleave_kernel<eT, 5 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 4 ) concat_n16cx_interleave_kernel<eT, 4 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 3 ) concat_n16cx_interleave_kernel<eT, 3 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 2 ) concat_n16cx_interleave_kernel<eT, 2 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 1 ) concat_n16cx_interleave_kernel<eT, 1 >(base_src, inner_start, inner_end, base_dst);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode concat_n16cx(
    const ppl::nn::TensorShape **src_shape_list,
    const eT **src_list,
    const int32_t num_src,
    const int32_t axis,
    eT *dst)
{
    const int32_t ndims      = int32_t(src_shape_list[0]->GetDimCount());
    const int32_t fixed_axis = axis < 0 ? ndims + axis : axis;
    const int32_t c_dim_idx  = 1;
    const int64_t c_blk      = 16;

    if (fixed_axis == c_dim_idx) { // concat on C dim
        for (int32_t i = 0; i < num_src - 1; i++) {
            if (src_shape_list[i]->GetDim(c_dim_idx) % c_blk != 0) {
                return concat_n16cx_interleave_channels<eT>(
                    src_shape_list,
                    src_list,
                    num_src,
                    axis,
                    c_dim_idx,
                    dst);
            }
        }
    }

    int64_t outer_dim = 1;
    int64_t inner_dim = c_blk;
    for (int32_t i = 0; i < fixed_axis; ++i) {
        if (i == c_dim_idx) {
            outer_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            outer_dim *= src_shape_list[0]->GetDim(i);
        }
    }
    for (int32_t i = fixed_axis + 1; i < ndims; ++i) {
        if (i == c_dim_idx) {
            inner_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            inner_dim *= src_shape_list[0]->GetDim(i);
        }
    }

    std::vector<int64_t> dst_offset(num_src);
    int64_t dst_concat_dim = 0;
    dst_offset[0]          = 0;
    if (fixed_axis == c_dim_idx) {
        for (int32_t i = 1; i < num_src; ++i) {
            dst_offset[i] = dst_offset[i - 1] + div_up(src_shape_list[i - 1]->GetDim(fixed_axis), c_blk);
        }
        dst_concat_dim = dst_offset[num_src - 1] + div_up(src_shape_list[num_src - 1]->GetDim(fixed_axis), c_blk);
    } else {
        for (int32_t i = 1; i < num_src; ++i) {
            dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(fixed_axis);
        }
        dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(fixed_axis);
    }

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (num_src * outer_dim >= num_threads) {
        if (fixed_axis == c_dim_idx) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                           src_list[n] + i * div_up(src_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dim,
                           div_up(src_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dim * sizeof(eT));
                }
            }
        } else {
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                           src_list[n] + i * src_shape_list[n]->GetDim(fixed_axis) * inner_dim,
                           src_shape_list[n]->GetDim(fixed_axis) * inner_dim * sizeof(eT));
                }
            }
        }
    } else { // parallel divide along concat dim, may across different src
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), dst_concat_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t concat_dim_per_thread = div_up(dst_concat_dim, num_threads);
            const int64_t start_concat_dim      = concat_dim_per_thread * thread_id;
            const int64_t end_concat_dim =      min(concat_dim_per_thread * (thread_id + 1), dst_concat_dim);

            if (start_concat_dim < end_concat_dim) {
                int64_t start_blk_idx = num_src - 1;
                int64_t end_blk_idx   = num_src - 1;
                for (int64_t i = 0; i < num_src - 1; i++) {
                    if (start_concat_dim >= dst_offset[i] && start_concat_dim < dst_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_concat_dim >= dst_offset[i] && end_concat_dim < dst_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_concat_dim - dst_offset[start_blk_idx];
                int64_t end_axis_idx   = end_concat_dim - dst_offset[end_blk_idx];

                if (fixed_axis == c_dim_idx) {
                    std::vector<concat_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy from single src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy from multiple src
                        infos.emplace_back(concat_parallel_info(
                            start_blk_idx, start_axis_idx, div_up(src_shape_list[start_blk_idx]->GetDim(fixed_axis), c_blk))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(concat_parallel_info(i, 0, div_up(src_shape_list[i]->GetDim(fixed_axis), c_blk))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dim; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const concat_parallel_info& info = infos[j];

                            const eT *p_src = src_list[info.blk_idx] + (i * div_up(src_shape_list[info.blk_idx]->GetDim(fixed_axis), c_blk) + info.start) * inner_dim;
                            eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                            const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                } else {
                    std::vector<concat_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy from single src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy from multiple src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, src_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(concat_parallel_info(i, 0, src_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dim; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const concat_parallel_info& info = infos[j];

                            const eT *p_src = src_list[info.blk_idx] + (i * src_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dim;
                            eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                            const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_CONCAT_CONCAT_COMMON_H_
