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
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/fp32/conv_transpose/col2im_fp32.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t conv_transpose_ndarray_fp32_get_buffer_bytes(
    const ppl::common::isa_t isa,
    const int64_t group,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t num_output,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w)
{
    (void) isa;
    const int64_t oc_per_grp = num_output / group;
    const bool do_col2im     = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                                pad_w == 0 && stride_h == 1 && stride_w == 1);
    const uint64_t col2im_len = !do_col2im ? 0 : uint64_t(oc_per_grp) * kernel_h * kernel_w * src_h * src_w;
    return col2im_len * sizeof(float);
}

ppl::common::RetCode conv_transpose_ndarray_fp32(
    const ppl::common::isa_t isa,
    const float *input,
    const float *filter,
    const float *bias,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t batch,
    const int64_t group,
    const int64_t channels,
    const int64_t num_output,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t hole_h,
    const int64_t hole_w,
    void *tmp_buffer,
    float *output)
{
    const int64_t ic_per_grp = channels / group;
    const int64_t oc_per_grp = num_output / group;
    const int64_t M = oc_per_grp * kernel_h * kernel_w;
    const int64_t N = src_h * src_w;
    const int64_t K = ic_per_grp;

    const int64_t lda   = M;
    const int64_t ldb   = N;
    const int64_t ldout = N;

    float *col2im_buffer = (float*)tmp_buffer;
    const bool do_col2im = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                            pad_w == 0 && stride_h == 1 && stride_w == 1);
    const bool gemm_with_bias = !do_col2im && bias;

    auto col2im_func = (isa & ppl::common::ISA_X86_AVX) ? col2im2d_ndarray_fp32_avx : col2im2d_ndarray_fp32_sse;

    for (int64_t g = 0; g < group; ++g) {
        const float *l_flt = filter + g * oc_per_grp * ic_per_grp * kernel_h * kernel_w;
        const float *l_bias = bias ? bias + g * oc_per_grp : nullptr;
        for (int64_t b = 0; b < batch; ++b) {  
            const float *l_src = input + (b * channels + g * ic_per_grp) * src_h * src_w;
            float *l_dst       = output + (b * num_output + g * oc_per_grp) * dst_h * dst_w;
            float *l_col       = do_col2im ? col2im_buffer : l_dst;

            auto ret = gemm_fp32(
                isa, l_flt, l_src, l_bias, nullptr,
                gemm_m_type::TRANS, gemm_m_type::NOTRANS,
                gemm_with_bias ? gemm_v_type::COL_VEC : gemm_v_type::EMPTY,
                gemm_m_type::EMPTY,
                M, N, K, lda, ldb, ldout, 0,
                1.0, 0.0, 1.0, 0.0, gemm_post::NONE, l_col);
            if (ppl::common::RC_SUCCESS != ret) {
                return ret;
            }

            if (do_col2im) {
                col2im_func(
                    l_col, l_bias, src_h, src_w, oc_per_grp,
                    dst_h, dst_w, kernel_h, kernel_w,
                    pad_h, pad_w, stride_h, stride_w,
                    hole_h, hole_w, l_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
