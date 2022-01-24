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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_DIRECT_NDARRAY_CONV_DIRECT_NDARRAY_H1W1_KERNEL_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_DIRECT_NDARRAY_CONV_DIRECT_NDARRAY_H1W1_KERNEL_H_

#ifdef PPLNN_USE_ARMV8_2_FP16

#include <arm_neon.h>
#include <cstdlib>
#include <iostream>

#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define CBLK() 8

template <const int64_t oc_section>
void ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    const int64_t src_hw_stride,
    const int64_t channels,
    const int64_t flt_h_start,
    const int64_t flt_h_end,
    const int64_t flt_w_start,
    const int64_t flt_w_end,
    const int64_t flt_w,
    const int64_t flt_ic_stride,
    const int64_t src_kh_stride,
    const int64_t dltn_w,
    const int64_t dst_bchw_stride,
    const uint32_t fuse_type);

template <>
void ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel<16>(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    const int64_t src_hw_stride,
    const int64_t channels,
    const int64_t flt_h_start,
    const int64_t flt_h_end,
    const int64_t flt_w_start,
    const int64_t flt_w_end,
    const int64_t flt_w,
    const int64_t flt_ic_stride,
    const int64_t src_kh_stride,
    const int64_t dltn_w,
    const int64_t dst_bchw_stride,
    const uint32_t fuse_type)
{
    float16x8_t v0, v1, v2, v3, v4, v5;

    if (bias_base == nullptr) {
        v0 = vld1q_f16(output_base);
        v1 = vld1q_f16(output_base + dst_bchw_stride);
    } else {
        v0 = vld1q_f16(bias_base);
        v1 = vld1q_f16(bias_base + CBLK());
    }

    int64_t ic                   = channels;
    const __fp16 *flt_ic_base = filter_base;
    const __fp16 *input_ic_base  = input_base;
    do {
        for (int64_t kh = flt_h_start; kh < flt_h_end; kh++) {
            const __fp16 *src_kh_base = input_ic_base + kh * src_kh_stride;
            for (int64_t kw = flt_w_start; kw < flt_w_end; kw++) {
                const __fp16 *filter_ptr = flt_ic_base + (kh * flt_w + kw) * CBLK() * 2;

                v2 = vld1q_f16(filter_ptr);
                v3 = vld1q_f16(filter_ptr + CBLK());
                v4 = vld1q_dup_f16(src_kh_base + kw * dltn_w);

                v0 = vfmaq_f16(v0, v2, v4);
                v1 = vfmaq_f16(v1, v3, v4);
            }
        }
        flt_ic_base += flt_ic_stride; //  flt_h * flt_w * CBLK() * 2;
        input_ic_base += src_hw_stride;
        ic -= 1;
    } while (ic > 0);

    if (fuse_type & conv_fuse_flag::SUM) { // sum
        v0 = vaddq_f16(v0, vld1q_f16(sum_base));
        v1 = vaddq_f16(v1, vld1q_f16(sum_base + dst_bchw_stride));
    }
    if (fuse_type & conv_fuse_flag::RELU) { // relu
        v5 = vdupq_n_f16(0.0);
        v0 = vmaxq_f16(v0, v5);
        v1 = vmaxq_f16(v1, v5);
    }
    if (fuse_type & conv_fuse_flag::RELU6) { // relu6
        v5 = vdupq_n_f16(6.0);
        v0 = vminq_f16(v0, v5);
        v1 = vminq_f16(v1, v5);
    }
    vst1q_f16(output_base, v0);
    vst1q_f16(output_base + dst_bchw_stride, v1);
}

template <>
void ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel<8>(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    const int64_t src_hw_stride,
    const int64_t channels,
    const int64_t flt_h_start,
    const int64_t flt_h_end,
    const int64_t flt_w_start,
    const int64_t flt_w_end,
    const int64_t flt_w,
    const int64_t flt_ic_stride,
    const int64_t src_kh_stride,
    const int64_t dltn_w,
    const int64_t dst_bchw_stride,
    const uint32_t fuse_type)
{
    float16x8_t v0, v2, v4, v5;

    if (bias_base == nullptr) {
        v0 = vld1q_f16(output_base);
    } else {
        v0 = vld1q_f16(bias_base);
    }

    int64_t ic                   = channels;
    const __fp16 *flt_ic_base = filter_base;
    const __fp16 *input_ic_base  = input_base;
    do {
        for (int64_t kh = flt_h_start; kh < flt_h_end; kh++) {
            const __fp16 *src_kh_base = input_ic_base + kh * src_kh_stride;
            for (int64_t kw = flt_w_start; kw < flt_w_end; kw++) {
                const __fp16 *filter_ptr = flt_ic_base + (kh * flt_w + kw) * CBLK() * 2;

                v2 = vld1q_f16(filter_ptr);
                v4 = vld1q_dup_f16(src_kh_base + kw * dltn_w);

                v0 = vfmaq_f16(v0, v2, v4);
            }
        }
        flt_ic_base += flt_ic_stride; //  flt_h * flt_w * CBLK() * 2;
        input_ic_base += src_hw_stride;
        ic -= 1;
    } while (ic > 0);

    if (fuse_type & conv_fuse_flag::SUM) { // sum
        v0 = vaddq_f16(v0, vld1q_f16(sum_base));
    }
    if (fuse_type & conv_fuse_flag::RELU) { // relu
        v5 = vdupq_n_f16(0.0);
        v0 = vmaxq_f16(v0, v5);
    }
    if (fuse_type & conv_fuse_flag::RELU6) { // relu6
        v5 = vdupq_n_f16(6.0);
        v0 = vminq_f16(v0, v5);
    }
    vst1q_f16(output_base, v0);
}

typedef void (*ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel_func_t)(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    const int64_t src_hw_stride,
    const int64_t channels,
    const int64_t flt_h_start,
    const int64_t flt_h_end,
    const int64_t flt_w_start,
    const int64_t flt_w_end,
    const int64_t flt_w,
    const int64_t flt_ic_stride,
    const int64_t src_kh_stride,
    const int64_t dltn_w,
    const int64_t dst_bchw_stride,
    const uint32_t fuse_type);

}}}}; // namespace ppl::kernel::arm_server::neon

#endif

#endif
