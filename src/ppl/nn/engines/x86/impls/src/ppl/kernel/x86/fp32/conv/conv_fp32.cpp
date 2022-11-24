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
#include "ppl/kernel/x86/common/conv_common.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t conv2d_ndarray_fp32_get_buffer_bytes(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t group,
    const int64_t channels,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w)
{
    (void) isa;
    const int64_t dst_h      = dst_shape->GetDim(2);
    const int64_t dst_w      = dst_shape->GetDim(3);
    const int64_t ic_per_grp = channels / group;
    const bool do_im2col     = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                                pad_w == 0 && stride_h == 1 && stride_w == 1);
    const uint64_t im2col_len = !do_im2col ? 0 : uint64_t(ic_per_grp) * kernel_h * kernel_w * dst_h * dst_w;
    return im2col_len * sizeof(float);
}

ppl::common::RetCode conv2d_ndarray_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *sum_src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float *sum_src,
    const float *filter,
    const float *bias,
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
    const conv_fuse_flag_t fuse_flag,
    void *temp_buffer,
    float *dst)
{
    const int64_t batch      = src_shape->GetDim(0);
    const int64_t src_c      = src_shape->GetDim(1);
    const int64_t src_h      = src_shape->GetDim(2);
    const int64_t src_w      = src_shape->GetDim(3);
    const int64_t dst_c      = dst_shape->GetDim(1);
    const int64_t dst_h      = dst_shape->GetDim(2);
    const int64_t dst_w      = dst_shape->GetDim(3);
    const int64_t sum_src_c  = sum_src_shape ? sum_src_shape->GetDim(1) : 0;
    const int64_t ic_per_grp = channels / group;
    const int64_t oc_per_grp = num_output / group;

    gemm_post_t post = gemm_post::NONE;
    gemm_m_type_t typesum = gemm_m_type::EMPTY;
    if (fuse_flag & conv_fuse_flag::RELU) {
        post |= gemm_post::RELU;
    }
    if (fuse_flag & conv_fuse_flag::RELU6) {
        post |= gemm_post::RELU6;
    }
    if (fuse_flag & conv_fuse_flag::SUM) {
        typesum = gemm_m_type::NOTRANS;
    }

    const int64_t M      = oc_per_grp;
    const int64_t N      = dst_h * dst_w;
    const int64_t K      = ic_per_grp * kernel_h * kernel_w;

    const int64_t lda   = K;
    const int64_t ldb   = N;
    const int64_t ldout = N;

    const bool do_im2col = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);
    float *im2col_buffer = (float*)temp_buffer;

    auto im2col_func = (isa & ppl::common::ISA_X86_AVX) ? im2col2d_ndarray_fp32_avx : im2col2d_ndarray_fp32_sse;

    for (int64_t g = 0; g < group; ++g) {
        auto l_flt = filter + g * oc_per_grp * ic_per_grp * kernel_h * kernel_w;
        auto l_bias = bias ? bias + g * oc_per_grp : nullptr;
        for (int64_t b = 0; b < batch; ++b) {  
            auto l_src = src + (b * src_c + g * ic_per_grp) * src_h * src_w;
            auto l_dst       = dst + (b * dst_c + g * oc_per_grp) * dst_h * dst_w;
            auto l_col       = do_im2col ? im2col_buffer : const_cast<float*>(l_src);
            auto l_sum       = sum_src + (b * sum_src_c + g * oc_per_grp) * dst_h * dst_w;

            if (do_im2col) {
                im2col_func(
                    l_src, ic_per_grp, src_h, src_w,
                    dst_h, dst_w, kernel_h, kernel_w,
                    pad_h, pad_w, stride_h, stride_w,
                    hole_h, hole_w, l_col);
            }

            auto ret = gemm_fp32(
                isa, l_flt, l_col, l_bias, l_sum,
                gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
                bias ? gemm_v_type::EMPTY : gemm_v_type::COL_VEC,
                typesum, M, N, K, lda, ldb, ldout, 0,
                1.0, 0.0, 1.0, 1.0, post, l_dst);
            if (ppl::common::RC_SUCCESS != ret) {
                return ret;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

uint64_t conv1d_ndarray_fp32_get_buffer_bytes(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t group,
    const int64_t channels,
    const int64_t kernel_w,
    const int64_t stride_w,
    const int64_t pad_w)
{
    const int64_t kernel_h = 1;
    const int64_t stride_h = 1;
    const int64_t pad_h    = 0;
    const int64_t dst_h    = 1;

    (void) isa;
    const int64_t dst_w      = dst_shape->GetDim(2);
    const int64_t ic_per_grp = channels / group;
    const bool do_im2col     = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                                pad_w == 0 && stride_h == 1 && stride_w == 1);
    const uint64_t im2col_len = !do_im2col ? 0 : uint64_t(ic_per_grp) * kernel_h * kernel_w * dst_h * dst_w;
    return im2col_len * sizeof(float);
}

ppl::common::RetCode conv1d_ndarray_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *sum_src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float *sum_src,
    const float *filter,
    const float *bias,
    const int64_t group,
    const int64_t channels,
    const int64_t num_output,
    const int64_t kernel_w,
    const int64_t stride_w,
    const int64_t pad_w,
    const int64_t hole_w,
    const conv_fuse_flag_t fuse_flag,
    void *temp_buffer,
    float *dst)
{
    const int64_t kernel_h = 1;
    const int64_t stride_h = 1;
    const int64_t pad_h    = 0;
    const int64_t hole_h   = 1;
    const int64_t src_h    = 1;
    const int64_t dst_h    = 1;

    const int64_t batch      = src_shape->GetDim(0);
    const int64_t src_c      = src_shape->GetDim(1);
    const int64_t src_w      = src_shape->GetDim(2);
    const int64_t dst_c      = dst_shape->GetDim(1);
    const int64_t dst_w      = dst_shape->GetDim(2);
    const int64_t sum_src_c  = sum_src_shape ? sum_src_shape->GetDim(1) : 0;
    const int64_t ic_per_grp = channels / group;
    const int64_t oc_per_grp = num_output / group;

    gemm_post_t post = gemm_post::NONE;
    gemm_m_type_t typesum = gemm_m_type::EMPTY;
    if (fuse_flag & conv_fuse_flag::RELU) {
        post |= gemm_post::RELU;
    }
    if (fuse_flag & conv_fuse_flag::RELU6) {
        post |= gemm_post::RELU6;
    }
    if (fuse_flag & conv_fuse_flag::SUM) {
        typesum = gemm_m_type::NOTRANS;
    }

    const int64_t M      = oc_per_grp;
    const int64_t N      = dst_h * dst_w;
    const int64_t K      = ic_per_grp * kernel_h * kernel_w;

    const int64_t lda   = K;
    const int64_t ldb   = N;
    const int64_t ldout = N;

    const bool do_im2col = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);
    float *im2col_buffer = (float*)temp_buffer;

    auto im2col_func = (isa & ppl::common::ISA_X86_AVX) ? im2col2d_ndarray_fp32_avx : im2col2d_ndarray_fp32_sse;

    for (int64_t g = 0; g < group; ++g) {
        auto l_flt = filter + g * oc_per_grp * ic_per_grp * kernel_h * kernel_w;
        auto l_bias = bias ? bias + g * oc_per_grp : nullptr;
        for (int64_t b = 0; b < batch; ++b) {  
            auto l_src = src + (b * src_c + g * ic_per_grp) * src_h * src_w;
            auto l_dst       = dst + (b * dst_c + g * oc_per_grp) * dst_h * dst_w;
            auto l_col       = do_im2col ? im2col_buffer : const_cast<float*>(l_src);
            auto l_sum       = sum_src + (b * sum_src_c + g * oc_per_grp) * dst_h * dst_w;

            if (do_im2col) {
                im2col_func(
                    l_src, ic_per_grp, src_h, src_w,
                    dst_h, dst_w, kernel_h, kernel_w,
                    pad_h, pad_w, stride_h, stride_w,
                    hole_h, hole_w, l_col);
            }

            auto ret = gemm_fp32(
                isa, l_flt, l_col, l_bias, l_sum,
                gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
                bias ? gemm_v_type::EMPTY : gemm_v_type::COL_VEC,
                typesum, M, N, K, lda, ldb, ldout, 0,
                1.0, 0.0, 1.0, 1.0, post, l_dst);
            if (ppl::common::RC_SUCCESS != ret) {
                return ret;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
