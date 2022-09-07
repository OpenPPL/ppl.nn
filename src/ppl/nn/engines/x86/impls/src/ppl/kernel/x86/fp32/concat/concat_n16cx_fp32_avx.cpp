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

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/common/concat/concat_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode concat_n16cx_interleave_channels_fp32_avx(
    const ppl::nn::TensorShape **src_shape_list,
    const float **src_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx,
    float *dst)
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
    const int64_t dst_channels       = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(c_dim_idx);
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
                float *base_dst           = dst + i * padded_oc * inner_dim + oc * inner_dim;
                const float *base_src[16] = {0};

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
                    memcpy32_avx(base_dst + inner_start * c_blk, base_src[0] + inner_start * c_blk, (inner_end - inner_start) * c_blk);
                    continue;
                }

                if (base_src[15] != 0 && ic_num == 1) {
                    const int32_t c_offset = c_blk - (first_ic % c_blk);
                    for (int64_t l = inner_start; l < inner_end; l++) {
                        memcpy32_avx(base_dst + l * c_blk,            base_src[0]        + l * c_blk, c_offset);
                        memcpy32_avx(base_dst + l * c_blk + c_offset, base_src[c_offset] + l * c_blk, c_blk - c_offset);
                    }
                    continue;
                }

                if      (oc_len_eff == 16) concat_n16cx_interleave_kernel<float, 16>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 15) concat_n16cx_interleave_kernel<float, 15>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 14) concat_n16cx_interleave_kernel<float, 14>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 13) concat_n16cx_interleave_kernel<float, 13>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 12) concat_n16cx_interleave_kernel<float, 12>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 11) concat_n16cx_interleave_kernel<float, 11>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 10) concat_n16cx_interleave_kernel<float, 10>(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 9 ) concat_n16cx_interleave_kernel<float, 9 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 8 ) concat_n16cx_interleave_kernel<float, 8 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 7 ) concat_n16cx_interleave_kernel<float, 7 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 6 ) concat_n16cx_interleave_kernel<float, 6 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 5 ) concat_n16cx_interleave_kernel<float, 5 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 4 ) concat_n16cx_interleave_kernel<float, 4 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 3 ) concat_n16cx_interleave_kernel<float, 3 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 2 ) concat_n16cx_interleave_kernel<float, 2 >(base_src, inner_start, inner_end, base_dst);
                else if (oc_len_eff == 1 ) concat_n16cx_interleave_kernel<float, 1 >(base_src, inner_start, inner_end, base_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
