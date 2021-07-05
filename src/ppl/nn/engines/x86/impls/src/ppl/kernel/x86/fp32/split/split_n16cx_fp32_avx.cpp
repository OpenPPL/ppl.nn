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

#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/avx_tools.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode split_n16cx_interleave_channels_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const float *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    const int32_t c_dim_idx,
    float **dst_list)
{
    const int32_t ndims = src_shape->GetDimCount();

    int64_t outer_dims = 1;
    int64_t inner_dims = 1;
    for (int64_t i = 0; i < c_dim_idx; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    for (int64_t i = c_dim_idx + 1; i < ndims; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0] = 0;
    for (int64_t i = 1; i < num_dst; i++) {
        src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t c_blk        = 16;
    const int64_t src_channels = src_shape->GetDim(c_dim_idx);
    const int64_t padded_ic    = round_up(src_channels, c_blk);

    const int64_t num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), inner_dims);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        const int64_t inner_dims_per_thread = div_up(inner_dims, num_threads);
        const int64_t start_inner_dims      = inner_dims_per_thread * thread_id;
        const int64_t end_inner_dims        = min(start_inner_dims + inner_dims_per_thread, inner_dims);

        if (start_inner_dims < end_inner_dims) {
            for (int64_t i = 0; i < outer_dims; i++) {
                for (int64_t n = 0; n < num_dst; n++) {
                    const int64_t dst_channels = dst_shape_list[n]->GetDim(c_dim_idx);
                    const int64_t padded_oc    = round_up(dst_channels, c_blk);
                    for (int64_t oc = 0; oc < padded_oc; oc += c_blk) {
                        const int64_t ic   = src_offset[n] + oc;
                        const float *p_src = src + i * padded_ic * inner_dims + round(ic, c_blk) * inner_dims;
                        float *p_dst       = dst_list[n] + i * padded_oc * inner_dims + oc * inner_dims;
                        if (ic % c_blk == 0) { // no interleave on this 16c
                            memcpy(p_dst + start_inner_dims * c_blk, p_src + start_inner_dims * c_blk, (end_inner_dims - start_inner_dims) * c_blk * sizeof(float));
                        } else { // has interleave on this 16c
                            const int64_t c_offset      = c_blk - (ic % c_blk);
                            const int64_t c_end         = min<int64_t>(dst_channels - oc, (int64_t)c_blk);
                            const int64_t c_copy_0      = c_offset;
                            const int64_t c_copy_1      = max<int64_t>(c_end - c_offset, 0);
                            const float *p_src_next_16c = p_src + c_blk * inner_dims;

                            if (oc + c_blk == padded_oc && dst_channels < padded_oc) { // last 16c need to pad 0
                                for (int64_t id = start_inner_dims; id < end_inner_dims; id++) {
                                    // interleave copy
                                    memcpy32_avx(p_dst + id * c_blk, p_src + id * c_blk + c_blk - c_offset, c_copy_0);
                                    memcpy32_avx(p_dst + id * c_blk + c_offset, p_src_next_16c + id * c_blk, c_copy_1);
                                }
                            } else {
                                for (int64_t id = start_inner_dims; id < end_inner_dims; id++) {
                                    // interleave copy
                                    memcpy32_avx(p_dst + id * c_blk, p_src + id * c_blk + c_blk - c_offset, c_copy_0);
                                    memcpy32_avx(p_dst + id * c_blk + c_offset, p_src_next_16c + id * c_blk, c_copy_1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
