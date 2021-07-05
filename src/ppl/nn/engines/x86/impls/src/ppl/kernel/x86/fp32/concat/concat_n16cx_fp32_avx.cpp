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

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0]       = 0;
    for (int32_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t c_blk        = 16;
    const int64_t dst_channels = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(c_dim_idx);
    const int64_t padded_oc    = round_up(dst_channels, c_blk);

    const int64_t num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), inner_dim);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        const int64_t inner_dim_per_thread = div_up(inner_dim, num_threads);
        const int64_t start_inner_dim      = inner_dim_per_thread * thread_id;
        const int64_t end_inner_dim        = min(inner_dim_per_thread * (thread_id + 1), inner_dim);

        if (start_inner_dim < end_inner_dim) {
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int32_t n = 0; n < num_src; n++) {
                    const int32_t src_channels = src_shape_list[n]->GetDim(c_dim_idx);
                    const int32_t padded_ic    = round_up(src_channels, c_blk);
                    for (int32_t ic = 0; ic < padded_ic; ic += c_blk) {
                        const int32_t oc   = dst_offset[n] + ic;
                        const float* p_src = src_list[n] + i * padded_ic * inner_dim + ic * inner_dim;
                        float* p_dst       = dst + i * padded_oc * inner_dim + round(oc, c_blk) * inner_dim;
                        if (oc % c_blk == 0) { // no interleave on this 16c
                            memcpy32_avx(p_dst + start_inner_dim * c_blk, p_src + start_inner_dim * c_blk, (end_inner_dim - start_inner_dim) * c_blk);
                        } else { // has interleave on this 16c
                            const int32_t c_offset = c_blk - (oc % c_blk);
                            const int32_t c_end    = min(src_channels - ic, (int32_t)c_blk);
                            const int32_t c_copy_0 = c_offset;
                            const int32_t c_copy_1 = max(c_end - c_offset, 0);
                            float *p_dst_next_16c  = p_dst + c_blk * inner_dim;

                            for (int64_t id = start_inner_dim; id < end_inner_dim; id++) {
                                // interleave copy
                                memcpy32_avx(p_dst + id * c_blk + c_blk - c_offset, p_src + id * c_blk, c_copy_0);
                                memcpy32_avx(p_dst_next_16c + id * c_blk, p_src + id * c_blk + c_offset, c_copy_1);
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
