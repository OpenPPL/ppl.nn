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
#include "ppl/kernel/x86/fp32/conv/im2col_fp32.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t conv_ndarray_fp32_get_buffer_bytes(
    const ppl::common::isa_t isa,
    const int64_t group,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t channels,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w)
{
    (void) isa;
    const int64_t ic_per_grp = channels / group;
    const bool do_im2col     = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                                pad_w == 0 && stride_h == 1 && stride_w == 1);
    const uint64_t im2col_len = !do_im2col ? 0 : uint64_t(ic_per_grp) * kernel_h * kernel_w * dst_h * dst_w;
    return im2col_len * sizeof(float);
}

ppl::common::RetCode conv_ndarray_fp32(
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
    const int64_t M      = oc_per_grp;
    const int64_t N      = dst_h * dst_w;
    const int64_t K      = ic_per_grp * kernel_h * kernel_w;

    const int64_t lda   = K;
    const int64_t ldb   = N;
    const int64_t ldout = N;

    const bool do_im2col = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);
    float *im2col_buffer = (float*)tmp_buffer;

    auto im2col_func = (isa & ppl::common::ISA_X86_AVX) ? im2col2d_ndarray_fp32_avx : im2col2d_ndarray_fp32_sse;

    for (int64_t g = 0; g < group; ++g) {
        const float *l_flt = filter + g * oc_per_grp * ic_per_grp * kernel_h * kernel_w;
        const float *l_bias = bias ? bias + g * oc_per_grp : nullptr;
        for (int64_t b = 0; b < batch; ++b) {  
            const float *l_src = input + (b * channels + g * ic_per_grp) * src_h * src_w;
            float *l_dst       = output + (b * num_output + g * oc_per_grp) * dst_h * dst_w;
            float *l_col       = do_im2col ? im2col_buffer : const_cast<float*>(l_src);

            if (do_im2col) {
                im2col_func(
                    l_src, ic_per_grp, src_h, src_w,
                    dst_h, dst_w, kernel_h, kernel_w,
                    pad_h, pad_w, stride_h, stride_w,
                    hole_h, hole_w, l_col);
            }

            auto ret = gemm_fp32(
                isa, l_flt, l_col, l_bias, nullptr,
                gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
                bias ? gemm_v_type::EMPTY : gemm_v_type::COL_VEC,
                gemm_m_type::EMPTY,
                M, N, K, lda, ldb, ldout, 0,
                1.0, 0.0, 1.0, 0.0, gemm_post::NONE, l_dst);
            if (ppl::common::RC_SUCCESS != ret) {
                return ret;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
