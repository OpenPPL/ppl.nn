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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/depthwise/conv2d_n4cx_depthwise_fp32.h"

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define CBLK()  4
#define ICBLK() CBLK()
#define OCBLK() CBLK()

static inline void conv_n4cx_depthwise_general_h1w1_kernel(
    const float *cvt_filter_ptr,
    const float *input_ptr,
    float *output_ptr,
    float *sum_ptr,
    const float32x4_t vbias,
    const int64_t src_w,
    const int64_t flt_w,
    const int64_t ih_base,
    const int64_t iw_base,
    const int64_t flt_h_start,
    const int64_t flt_h_end,
    const int64_t flt_w_start,
    const int64_t flt_w_end,
    const int64_t dltn_h,
    const int64_t dltn_w,
    const uint32_t fuse_flag)
{
    float32x4_t vout = vbias;
    for (int64_t fh = flt_h_start; fh < flt_h_end; fh++) {
        const float *input_h_base  = input_ptr + fh * dltn_h * src_w * ICBLK();
        const float *filter_h_base = cvt_filter_ptr + fh * flt_w * CBLK();
        for (int64_t fw = flt_w_start; fw < flt_w_end; fw++) {
            float32x4_t vin  = vld1q_f32(input_h_base + fw * dltn_w * ICBLK());
            float32x4_t vflt = vld1q_f32(filter_h_base + fw * CBLK());
            vout             = vfmaq_f32(vout, vin, vflt);
        }
    }

    if (fuse_flag & conv_fuse_flag::SUM) {
        vout = vaddq_f32(vout, vld1q_f32(sum_ptr));
    }
    if (fuse_flag & conv_fuse_flag::RELU) {
        vout = vmaxq_f32(vout, vdupq_n_f32(0.0f));
    }
    if (fuse_flag & conv_fuse_flag::RELU6) {
        vout = vminq_f32(vout, vdupq_n_f32(6.0f));
    }
    vst1q_f32(output_ptr, vout);
}

static inline void conv_n4cx_depthwise_general_h1w8_kernel(
    const float *cvt_filter_ptr,
    const float *input_ptr,
    float *output_ptr,
    float *sum_ptr,
    const float32x4_t vbias,
    const int64_t flt_w,
    const int64_t strd_w,
    const int64_t ih_base,
    const int64_t iw_base,
    const int64_t flt_h_start,
    const int64_t flt_h_end,
    const int64_t dltn_h_x_src_w,
    const int64_t dltn_w,
    const uint32_t fuse_flag)
{
    float32x4_t vout0 = vbias;
    float32x4_t vout1 = vbias;
    float32x4_t vout2 = vbias;
    float32x4_t vout3 = vbias;
    float32x4_t vout4 = vbias;
    float32x4_t vout5 = vbias;
    float32x4_t vout6 = vbias;
    float32x4_t vout7 = vbias;

    for (int64_t fh = flt_h_start; fh < flt_h_end; fh++) {
        const float *filter_h_base = cvt_filter_ptr + fh * flt_w * CBLK();
        const float *input_h_base  = input_ptr + fh * dltn_h_x_src_w * ICBLK();
        for (int64_t fw = 0; fw < flt_w; fw++) {
            const float *input_base = input_h_base + fw * dltn_w * ICBLK();
            float32x4_t vflt        = vld1q_f32(filter_h_base + fw * CBLK());

            float32x4_t vin0 = vld1q_f32(input_base);
            float32x4_t vin1 = vld1q_f32(input_base + strd_w * OCBLK());
            float32x4_t vin2 = vld1q_f32(input_base + strd_w * OCBLK() * 2);
            float32x4_t vin3 = vld1q_f32(input_base + strd_w * OCBLK() * 3);
            float32x4_t vin4 = vld1q_f32(input_base + strd_w * OCBLK() * 4);
            float32x4_t vin5 = vld1q_f32(input_base + strd_w * OCBLK() * 5);
            float32x4_t vin6 = vld1q_f32(input_base + strd_w * OCBLK() * 6);
            float32x4_t vin7 = vld1q_f32(input_base + strd_w * OCBLK() * 7);

            vout0 = vfmaq_f32(vout0, vin0, vflt);
            vout1 = vfmaq_f32(vout1, vin1, vflt);
            vout2 = vfmaq_f32(vout2, vin2, vflt);
            vout3 = vfmaq_f32(vout3, vin3, vflt);
            vout4 = vfmaq_f32(vout4, vin4, vflt);
            vout5 = vfmaq_f32(vout5, vin5, vflt);
            vout6 = vfmaq_f32(vout6, vin6, vflt);
            vout7 = vfmaq_f32(vout7, vin7, vflt);
        }
    }
    if (fuse_flag & conv_fuse_flag::SUM) {
        vout0 = vaddq_f32(vout0, vld1q_f32(sum_ptr));
        vout1 = vaddq_f32(vout1, vld1q_f32(sum_ptr + OCBLK() * 1));
        vout2 = vaddq_f32(vout2, vld1q_f32(sum_ptr + OCBLK() * 2));
        vout3 = vaddq_f32(vout3, vld1q_f32(sum_ptr + OCBLK() * 3));
        vout4 = vaddq_f32(vout4, vld1q_f32(sum_ptr + OCBLK() * 4));
        vout5 = vaddq_f32(vout5, vld1q_f32(sum_ptr + OCBLK() * 5));
        vout6 = vaddq_f32(vout6, vld1q_f32(sum_ptr + OCBLK() * 6));
        vout7 = vaddq_f32(vout7, vld1q_f32(sum_ptr + OCBLK() * 7));
    }
    if (fuse_flag & conv_fuse_flag::RELU) {
        float32x4_t vzero = vdupq_n_f32(0.0f);
        vout0             = vmaxq_f32(vout0, vzero);
        vout1             = vmaxq_f32(vout1, vzero);
        vout2             = vmaxq_f32(vout2, vzero);
        vout3             = vmaxq_f32(vout3, vzero);
        vout4             = vmaxq_f32(vout4, vzero);
        vout5             = vmaxq_f32(vout5, vzero);
        vout6             = vmaxq_f32(vout6, vzero);
        vout7             = vmaxq_f32(vout7, vzero);
    }
    if (fuse_flag & conv_fuse_flag::RELU6) {
        float32x4_t vsix = vdupq_n_f32(6.0f);
        vout0            = vminq_f32(vout0, vsix);
        vout1            = vminq_f32(vout1, vsix);
        vout2            = vminq_f32(vout2, vsix);
        vout3            = vminq_f32(vout3, vsix);
        vout4            = vminq_f32(vout4, vsix);
        vout5            = vminq_f32(vout5, vsix);
        vout6            = vminq_f32(vout6, vsix);
        vout7            = vminq_f32(vout7, vsix);
    }
    vst1q_f32(output_ptr, vout0);
    vst1q_f32(output_ptr + OCBLK() * 1, vout1);
    vst1q_f32(output_ptr + OCBLK() * 2, vout2);
    vst1q_f32(output_ptr + OCBLK() * 3, vout3);
    vst1q_f32(output_ptr + OCBLK() * 4, vout4);
    vst1q_f32(output_ptr + OCBLK() * 5, vout5);
    vst1q_f32(output_ptr + OCBLK() * 6, vout6);
    vst1q_f32(output_ptr + OCBLK() * 7, vout7);
}

template <const uint32_t padding, const uint32_t stride>
void conv_n4cx_depthwise_f3sx_h1w4(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t fltC,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_batch,
    const uint32_t fuse_flag);

template <>
void conv_n4cx_depthwise_f3sx_h1w4<0, 1>(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t fltC,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    PRAGMA_OMP_PARALLEL()
    {
        int64_t dst_w_align4 = (dst_w & (~3));

        const int64_t fltC_pck            = CEIL4(fltC);
        const int64_t src_hW                = src_h * src_w;
        const int64_t dst_hW               = dst_h * dst_w;
        const int64_t input_batch_stride  = fltC_pck * src_hW;
        const int64_t output_batch_stride = fltC_pck * dst_hW;

        for (int64_t b = 0; b < num_batch; b++) {
            for (int64_t c = 0; c < fltC; c += CBLK()) {
                const float *converted_filter_c_base = converted_filter + c * 9;
                const float *bias_c_base             = bias + c;
                const float *input_c_base            = input + b * input_batch_stride + c * src_hW;
                float *output_c_base                 = output + b * output_batch_stride + c * dst_hW;
                float *sum_c_base                    = sum + b * output_batch_stride + c * dst_hW;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
                float32x4_t vflt[9];
                vflt[0]           = vld1q_f32(converted_filter_c_base + 0 * CBLK());
                vflt[1]           = vld1q_f32(converted_filter_c_base + 1 * CBLK());
                vflt[2]           = vld1q_f32(converted_filter_c_base + 2 * CBLK());
                vflt[3]           = vld1q_f32(converted_filter_c_base + 3 * CBLK());
                vflt[4]           = vld1q_f32(converted_filter_c_base + 4 * CBLK());
                vflt[5]           = vld1q_f32(converted_filter_c_base + 5 * CBLK());
                vflt[6]           = vld1q_f32(converted_filter_c_base + 6 * CBLK());
                vflt[7]           = vld1q_f32(converted_filter_c_base + 7 * CBLK());
                vflt[8]           = vld1q_f32(converted_filter_c_base + 8 * CBLK());
                float32x4_t vbias = vld1q_f32(bias_c_base);

                PRAGMA_OMP_FOR_NOWAIT()
                for (int64_t oh = 0; oh < dst_h; oh++) {
                    const float *input_h_base = input_c_base + oh * src_w * ICBLK();
                    float *output_h_base      = output_c_base + oh * dst_w * OCBLK();
                    float *sum_h_base         = sum_c_base + oh * dst_w * OCBLK();
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK(), 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK() * 2, 0, 3);

                    for (int64_t ow = 0; ow < dst_w_align4; ow += 4) {
                        const float *input_ptr = input_h_base + ow * ICBLK();
                        float *output_ptr      = output_h_base + ow * OCBLK();
                        __builtin_prefetch(output_ptr, 1, 2);

                        float32x4_t vin[18];
                        float32x4_t vout[4];

                        vout[0] = vbias;
                        vout[1] = vbias;
                        vout[2] = vbias;
                        vout[3] = vbias;

                        vin[0] = vld1q_f32(input_ptr);
                        vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);
                        vin[3] = vld1q_f32(input_ptr + ICBLK() * 3);
                        vin[4] = vld1q_f32(input_ptr + ICBLK() * 4);
                        vin[5] = vld1q_f32(input_ptr + ICBLK() * 5);

                        vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                        vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                        vin[8]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                        vin[9]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 3);
                        vin[10] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 4);
                        vin[11] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 5);

                        vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                        vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                        vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                        vin[15] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 3);
                        vin[16] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 4);
                        vin[17] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 5);

                        vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                        vout[1] = vfmaq_f32(vout[1], vin[1], vflt[0]);
                        vout[2] = vfmaq_f32(vout[2], vin[2], vflt[0]);
                        vout[3] = vfmaq_f32(vout[3], vin[3], vflt[0]);

                        vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                        vout[1] = vfmaq_f32(vout[1], vin[2], vflt[1]);
                        vout[2] = vfmaq_f32(vout[2], vin[3], vflt[1]);
                        vout[3] = vfmaq_f32(vout[3], vin[4], vflt[1]);

                        vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                        vout[1] = vfmaq_f32(vout[1], vin[3], vflt[2]);
                        vout[2] = vfmaq_f32(vout[2], vin[4], vflt[2]);
                        vout[3] = vfmaq_f32(vout[3], vin[5], vflt[2]);

                        vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                        vout[1] = vfmaq_f32(vout[1], vin[7], vflt[3]);
                        vout[2] = vfmaq_f32(vout[2], vin[8], vflt[3]);
                        vout[3] = vfmaq_f32(vout[3], vin[9], vflt[3]);

                        vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                        vout[1] = vfmaq_f32(vout[1], vin[8], vflt[4]);
                        vout[2] = vfmaq_f32(vout[2], vin[9], vflt[4]);
                        vout[3] = vfmaq_f32(vout[3], vin[10], vflt[4]);

                        vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                        vout[1] = vfmaq_f32(vout[1], vin[9], vflt[5]);
                        vout[2] = vfmaq_f32(vout[2], vin[10], vflt[5]);
                        vout[3] = vfmaq_f32(vout[3], vin[11], vflt[5]);

                        vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                        vout[1] = vfmaq_f32(vout[1], vin[13], vflt[6]);
                        vout[2] = vfmaq_f32(vout[2], vin[14], vflt[6]);
                        vout[3] = vfmaq_f32(vout[3], vin[15], vflt[6]);

                        vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                        vout[1] = vfmaq_f32(vout[1], vin[14], vflt[7]);
                        vout[2] = vfmaq_f32(vout[2], vin[15], vflt[7]);
                        vout[3] = vfmaq_f32(vout[3], vin[16], vflt[7]);

                        vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                        vout[1] = vfmaq_f32(vout[1], vin[15], vflt[8]);
                        vout[2] = vfmaq_f32(vout[2], vin[16], vflt[8]);
                        vout[3] = vfmaq_f32(vout[3], vin[17], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            float *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                            vout[1]        = vaddq_f32(vout[1], vld1q_f32(sum_ptr + OCBLK() * 1));
                            vout[2]        = vaddq_f32(vout[2], vld1q_f32(sum_ptr + OCBLK() * 2));
                            vout[3]        = vaddq_f32(vout[3], vld1q_f32(sum_ptr + OCBLK() * 3));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            float32x4_t vzero = vdupq_n_f32(0.0f);
                            vout[0]           = vmaxq_f32(vout[0], vzero);
                            vout[1]           = vmaxq_f32(vout[1], vzero);
                            vout[2]           = vmaxq_f32(vout[2], vzero);
                            vout[3]           = vmaxq_f32(vout[3], vzero);
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            float32x4_t vsix = vdupq_n_f32(6.0f);
                            vout[0]          = vminq_f32(vout[0], vsix);
                            vout[1]          = vminq_f32(vout[1], vsix);
                            vout[2]          = vminq_f32(vout[2], vsix);
                            vout[3]          = vminq_f32(vout[3], vsix);
                        }

                        vst1q_f32(output_ptr, vout[0]);
                        vst1q_f32(output_ptr + OCBLK() * 1, vout[1]);
                        vst1q_f32(output_ptr + OCBLK() * 2, vout[2]);
                        vst1q_f32(output_ptr + OCBLK() * 3, vout[3]);
                    }
                    for (int64_t ow = dst_w_align4; ow < dst_w; ow++) {
                        const float *input_ptr = input_h_base + ow * ICBLK();
                        float *output_ptr      = output_h_base + ow * OCBLK();

                        float32x4_t vin[15];
                        float32x4_t vout[1];

                        vout[0] = vbias;

                        vin[0] = vld1q_f32(input_ptr);
                        vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);

                        vin[6] = vld1q_f32(input_ptr + src_w * ICBLK());
                        vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                        vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);

                        vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                        vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                        vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);

                        vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                        vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                        vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                        vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                        vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                        vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                        vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                        vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                        vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            float *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                        }

                        vst1q_f32(output_ptr, vout[0]);
                    }
                }
#pragma GCC diagnostic pop
            }
        }
    }
}

template <>
void conv_n4cx_depthwise_f3sx_h1w4<1, 1>(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t fltC,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t oh_inner_start = 1; // inclusive index
        const int64_t ow_inner_start = 1; // inclusive index
        int64_t oh_inner_end         = src_h - 1; // exclusive index
        int64_t ow_inner_end         = src_w - 1; // exclusive index

        oh_inner_end = std::max(oh_inner_end, oh_inner_start);
        ow_inner_end = std::max(ow_inner_end, ow_inner_start);

        int64_t ow_inner_end_align4 = ((ow_inner_end - ow_inner_start) & (~3)) + ow_inner_start;

        const int64_t fltC_pck            = CEIL4(fltC);
        const int64_t src_hW                = src_h * src_w;
        const int64_t dst_hW               = dst_h * dst_w;
        const int64_t input_batch_stride  = fltC_pck * src_hW;
        const int64_t output_batch_stride = fltC_pck * dst_hW;

        int64_t p_c = -1;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        float32x4_t vflt[9];
        float32x4_t vbias;

        PRAGMA_OMP_FOR_COLLAPSE_NOWAIT(3)
        for (int64_t b = 0; b < num_batch; b++) {
            for (int64_t c = 0; c < fltC; c += CBLK()) {
                for (int64_t oh = 0; oh < dst_h; oh++) {
                    const float *converted_filter_c_base = converted_filter + c * 9;
                    const float *bias_c_base             = bias + c;
                    const float *input_c_base            = input + b * input_batch_stride + c * src_hW;
                    float *output_c_base                 = output + b * output_batch_stride + c * dst_hW;
                    float *sum_c_base                    = sum + b * output_batch_stride + c * dst_hW;

                    if (p_c != c) {
                        vflt[0] = vld1q_f32(converted_filter_c_base + 0 * CBLK());
                        vflt[1] = vld1q_f32(converted_filter_c_base + 1 * CBLK());
                        vflt[2] = vld1q_f32(converted_filter_c_base + 2 * CBLK());
                        vflt[3] = vld1q_f32(converted_filter_c_base + 3 * CBLK());
                        vflt[4] = vld1q_f32(converted_filter_c_base + 4 * CBLK());
                        vflt[5] = vld1q_f32(converted_filter_c_base + 5 * CBLK());
                        vflt[6] = vld1q_f32(converted_filter_c_base + 6 * CBLK());
                        vflt[7] = vld1q_f32(converted_filter_c_base + 7 * CBLK());
                        vflt[8] = vld1q_f32(converted_filter_c_base + 8 * CBLK());
                        vbias   = vld1q_f32(bias_c_base);
                        p_c     = c;
                    }

                    const float *input_h_base = input_c_base + (oh - 1) * src_w * ICBLK();
                    float *output_h_base      = output_c_base + oh * dst_w * OCBLK();
                    float *sum_h_base         = sum_c_base + oh * dst_w * OCBLK();
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK(), 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK() * 2, 0, 3);
                    __builtin_prefetch(output_h_base, 1, 2);

                    if (oh == 0 || oh == dst_h - 1) {
                        bool ih0_valid = (oh >= 1);
                        bool ih2_valid = (oh < src_h - 1);

                        {
                            float32x4_t vin[18];
                            float32x4_t vout[4];
                            __builtin_prefetch(output_h_base, 1, 2);
                            const float *input_ptr = input_h_base;
                            bool iw2_valid         = (1 < src_w);

                            vout[0] = vbias;

                            if (ih0_valid) {
                                vin[1]  = vld1q_f32(input_ptr);
                                vin[2]  = (iw2_valid) ? vld1q_f32(input_ptr + ICBLK()) : vdupq_n_f32(0.0f);
                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            }

                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[8]  = (iw2_valid) ? vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK()) : vdupq_n_f32(0.0f);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);

                            if (ih2_valid) {
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[14] = (iw2_valid) ? vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK()) : vdupq_n_f32(0.0f);
                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                                vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_h_base, vout[0]);
                        }
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow += 4) {
                            const float *input_ptr = input_h_base + (ow - 1) * ICBLK();
                            float *output_ptr      = output_h_base + ow * OCBLK();
                            __builtin_prefetch(output_ptr, 1, 2);

                            float32x4_t vin[18];
                            float32x4_t vout[4];

                            vout[0] = vbias;
                            vout[1] = vbias;
                            vout[2] = vbias;
                            vout[3] = vbias;
                            if (ih0_valid) {
                                vin[0] = vld1q_f32(input_ptr);
                                vin[1] = vld1q_f32(input_ptr + ICBLK());
                                vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);
                                vin[3] = vld1q_f32(input_ptr + ICBLK() * 3);
                                vin[4] = vld1q_f32(input_ptr + ICBLK() * 4);
                                vin[5] = vld1q_f32(input_ptr + ICBLK() * 5);

                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                                vout[1] = vfmaq_f32(vout[1], vin[1], vflt[0]);
                                vout[2] = vfmaq_f32(vout[2], vin[2], vflt[0]);
                                vout[3] = vfmaq_f32(vout[3], vin[3], vflt[0]);

                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                                vout[1] = vfmaq_f32(vout[1], vin[2], vflt[1]);
                                vout[2] = vfmaq_f32(vout[2], vin[3], vflt[1]);
                                vout[3] = vfmaq_f32(vout[3], vin[4], vflt[1]);

                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                                vout[1] = vfmaq_f32(vout[1], vin[3], vflt[2]);
                                vout[2] = vfmaq_f32(vout[2], vin[4], vflt[2]);
                                vout[3] = vfmaq_f32(vout[3], vin[5], vflt[2]);
                            }

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                            vin[8]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                            vin[9]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 3);
                            vin[10] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 4);
                            vin[11] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 5);

                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[1] = vfmaq_f32(vout[1], vin[7], vflt[3]);
                            vout[2] = vfmaq_f32(vout[2], vin[8], vflt[3]);
                            vout[3] = vfmaq_f32(vout[3], vin[9], vflt[3]);

                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[1] = vfmaq_f32(vout[1], vin[8], vflt[4]);
                            vout[2] = vfmaq_f32(vout[2], vin[9], vflt[4]);
                            vout[3] = vfmaq_f32(vout[3], vin[10], vflt[4]);

                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                            vout[1] = vfmaq_f32(vout[1], vin[9], vflt[5]);
                            vout[2] = vfmaq_f32(vout[2], vin[10], vflt[5]);
                            vout[3] = vfmaq_f32(vout[3], vin[11], vflt[5]);

                            if (ih2_valid) {
                                vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());
                                vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                                vin[15] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 3);
                                vin[16] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 4);
                                vin[17] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 5);

                                vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                                vout[1] = vfmaq_f32(vout[1], vin[13], vflt[6]);
                                vout[2] = vfmaq_f32(vout[2], vin[14], vflt[6]);
                                vout[3] = vfmaq_f32(vout[3], vin[15], vflt[6]);

                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                                vout[1] = vfmaq_f32(vout[1], vin[14], vflt[7]);
                                vout[2] = vfmaq_f32(vout[2], vin[15], vflt[7]);
                                vout[3] = vfmaq_f32(vout[3], vin[16], vflt[7]);

                                vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                                vout[1] = vfmaq_f32(vout[1], vin[15], vflt[8]);
                                vout[2] = vfmaq_f32(vout[2], vin[16], vflt[8]);
                                vout[3] = vfmaq_f32(vout[3], vin[17], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                                vout[1]        = vaddq_f32(vout[1], vld1q_f32(sum_ptr + OCBLK() * 1));
                                vout[2]        = vaddq_f32(vout[2], vld1q_f32(sum_ptr + OCBLK() * 2));
                                vout[3]        = vaddq_f32(vout[3], vld1q_f32(sum_ptr + OCBLK() * 3));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                float32x4_t vzero = vdupq_n_f32(0.0f);
                                vout[0]           = vmaxq_f32(vout[0], vzero);
                                vout[1]           = vmaxq_f32(vout[1], vzero);
                                vout[2]           = vmaxq_f32(vout[2], vzero);
                                vout[3]           = vmaxq_f32(vout[3], vzero);
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                float32x4_t vsix = vdupq_n_f32(6.0f);
                                vout[0]          = vminq_f32(vout[0], vsix);
                                vout[1]          = vminq_f32(vout[1], vsix);
                                vout[2]          = vminq_f32(vout[2], vsix);
                                vout[3]          = vminq_f32(vout[3], vsix);
                            }

                            vst1q_f32(output_ptr, vout[0]);
                            vst1q_f32(output_ptr + OCBLK(), vout[1]);
                            vst1q_f32(output_ptr + OCBLK() * 2, vout[2]);
                            vst1q_f32(output_ptr + OCBLK() * 3, vout[3]);
                        }
                        for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                            const float *input_ptr = input_h_base + (ow - 1) * ICBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[4];

                            vout[0] = vbias;
                            if (ih0_valid) {
                                vin[0]  = vld1q_f32(input_ptr);
                                vin[1]  = vld1q_f32(input_ptr + ICBLK());
                                vin[2]  = vld1q_f32(input_ptr + ICBLK() * 2);
                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            }

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());
                            vin[8]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);

                            if (ih2_valid) {
                                vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());
                                vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                                vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                                vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_h_base + ow * OCBLK(), vout[0]);
                        }
                        if (ow_inner_end < dst_w) {
                            const float *input_ptr = input_h_base + (src_w - 2) * ICBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;
                            if (ih0_valid) {
                                vin[0]  = vld1q_f32(input_ptr);
                                vin[1]  = vld1q_f32(input_ptr + ICBLK());
                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            }

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);

                            if (ih2_valid) {
                                vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());
                                vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base + (dst_w - 1) * OCBLK()));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_c_base + oh * dst_w * OCBLK() + (dst_w - 1) * OCBLK(), vout[0]);
                        }
                    } else {
                        {
                            __builtin_prefetch(output_h_base, 1, 2);
                            const float *input_ptr = input_h_base;
                            bool iw2_valid         = (1 < src_w);

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            vin[1]  = vld1q_f32(input_ptr);
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);

                            if (iw2_valid) {
                                vin[2]  = vld1q_f32(input_ptr + ICBLK());
                                vin[8]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());
                                vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());

                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                                vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                                vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_h_base, vout[0]);
                        }
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow += 4) {
                            int64_t iw             = -1 + ow;
                            const float *input_ptr = input_h_base + iw * ICBLK();
                            float *output_ptr      = output_h_base + ow * OCBLK();
                            __builtin_prefetch(output_ptr, 1, 2);

                            float32x4_t vin[18];
                            float32x4_t vout[4];

                            vout[0] = vbias;
                            vout[1] = vbias;
                            vout[2] = vbias;
                            vout[3] = vbias;

                            vin[0] = vld1q_f32(input_ptr);
                            vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                            vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);
                            vin[3] = vld1q_f32(input_ptr + ICBLK() * 3);
                            vin[4] = vld1q_f32(input_ptr + ICBLK() * 4);
                            vin[5] = vld1q_f32(input_ptr + ICBLK() * 5);

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                            vin[8]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                            vin[9]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 3);
                            vin[10] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 4);
                            vin[11] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 5);

                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                            vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                            vin[15] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 3);
                            vin[16] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 4);
                            vin[17] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 5);

                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                            vout[1] = vfmaq_f32(vout[1], vin[1], vflt[0]);
                            vout[2] = vfmaq_f32(vout[2], vin[2], vflt[0]);
                            vout[3] = vfmaq_f32(vout[3], vin[3], vflt[0]);

                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[1] = vfmaq_f32(vout[1], vin[2], vflt[1]);
                            vout[2] = vfmaq_f32(vout[2], vin[3], vflt[1]);
                            vout[3] = vfmaq_f32(vout[3], vin[4], vflt[1]);

                            vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            vout[1] = vfmaq_f32(vout[1], vin[3], vflt[2]);
                            vout[2] = vfmaq_f32(vout[2], vin[4], vflt[2]);
                            vout[3] = vfmaq_f32(vout[3], vin[5], vflt[2]);

                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[1] = vfmaq_f32(vout[1], vin[7], vflt[3]);
                            vout[2] = vfmaq_f32(vout[2], vin[8], vflt[3]);
                            vout[3] = vfmaq_f32(vout[3], vin[9], vflt[3]);

                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[1] = vfmaq_f32(vout[1], vin[8], vflt[4]);
                            vout[2] = vfmaq_f32(vout[2], vin[9], vflt[4]);
                            vout[3] = vfmaq_f32(vout[3], vin[10], vflt[4]);

                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                            vout[1] = vfmaq_f32(vout[1], vin[9], vflt[5]);
                            vout[2] = vfmaq_f32(vout[2], vin[10], vflt[5]);
                            vout[3] = vfmaq_f32(vout[3], vin[11], vflt[5]);

                            vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                            vout[1] = vfmaq_f32(vout[1], vin[13], vflt[6]);
                            vout[2] = vfmaq_f32(vout[2], vin[14], vflt[6]);
                            vout[3] = vfmaq_f32(vout[3], vin[15], vflt[6]);

                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                            vout[1] = vfmaq_f32(vout[1], vin[14], vflt[7]);
                            vout[2] = vfmaq_f32(vout[2], vin[15], vflt[7]);
                            vout[3] = vfmaq_f32(vout[3], vin[16], vflt[7]);

                            vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                            vout[1] = vfmaq_f32(vout[1], vin[15], vflt[8]);
                            vout[2] = vfmaq_f32(vout[2], vin[16], vflt[8]);
                            vout[3] = vfmaq_f32(vout[3], vin[17], vflt[8]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                                vout[1]        = vaddq_f32(vout[1], vld1q_f32(sum_ptr + OCBLK() * 1));
                                vout[2]        = vaddq_f32(vout[2], vld1q_f32(sum_ptr + OCBLK() * 2));
                                vout[3]        = vaddq_f32(vout[3], vld1q_f32(sum_ptr + OCBLK() * 3));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                float32x4_t vzero = vdupq_n_f32(0.0f);
                                vout[0]           = vmaxq_f32(vout[0], vzero);
                                vout[1]           = vmaxq_f32(vout[1], vzero);
                                vout[2]           = vmaxq_f32(vout[2], vzero);
                                vout[3]           = vmaxq_f32(vout[3], vzero);
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                float32x4_t vsix = vdupq_n_f32(6.0f);
                                vout[0]          = vminq_f32(vout[0], vsix);
                                vout[1]          = vminq_f32(vout[1], vsix);
                                vout[2]          = vminq_f32(vout[2], vsix);
                                vout[3]          = vminq_f32(vout[3], vsix);
                            }

                            vst1q_f32(output_ptr, vout[0]);
                            vst1q_f32(output_ptr + OCBLK() * 1, vout[1]);
                            vst1q_f32(output_ptr + OCBLK() * 2, vout[2]);
                            vst1q_f32(output_ptr + OCBLK() * 3, vout[3]);
                        }
                        for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                            int64_t iw             = -1 + ow;
                            const float *input_ptr = input_h_base + iw * ICBLK();
                            float *output_ptr      = output_h_base + ow * OCBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            vin[0] = vld1q_f32(input_ptr);
                            vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                            vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);

                            vin[6] = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                            vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);

                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                            vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);

                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                            vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_ptr, vout[0]);
                        }
                        if (ow_inner_end < dst_w) {
                            // ow = dst_w - 1 == src_w - 1 (as dst_w == src_w f3p1s1d1)
                            const float *input_ptr = input_h_base + (src_w - 2) * ICBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            vin[0] = vld1q_f32(input_ptr);
                            vin[1] = vld1q_f32(input_ptr + ICBLK());

                            vin[6] = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());

                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());

                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base + (dst_w - 1) * OCBLK()));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_c_base + oh * dst_w * OCBLK() + (dst_w - 1) * OCBLK(), vout[0]);
                        }
                    }
                }
            }
        }
#pragma GCC diagnostic pop
    }
}

template <>
void conv_n4cx_depthwise_f3sx_h1w4<0, 2>(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t fltC,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    PRAGMA_OMP_PARALLEL()
    {
        int64_t dst_w_align4 = (dst_w & (~3));

        const int64_t fltC_pck            = CEIL4(fltC);
        const int64_t src_hW                = src_h * src_w;
        const int64_t dst_hW               = dst_h * dst_w;
        const int64_t input_batch_stride  = fltC_pck * src_hW;
        const int64_t output_batch_stride = fltC_pck * dst_hW;

        for (int64_t b = 0; b < num_batch; b++) {
            for (int64_t c = 0; c < fltC; c += CBLK()) {
                const float *converted_filter_c_base = converted_filter + c * 9;
                const float *bias_c_base             = bias + c;
                const float *input_c_base            = input + b * input_batch_stride + c * src_hW;
                float *output_c_base                 = output + b * output_batch_stride + c * dst_hW;
                float *sum_c_base                    = sum + b * output_batch_stride + c * dst_hW;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
                float32x4_t vflt[9];
                vflt[0]           = vld1q_f32(converted_filter_c_base + 0 * CBLK());
                vflt[1]           = vld1q_f32(converted_filter_c_base + 1 * CBLK());
                vflt[2]           = vld1q_f32(converted_filter_c_base + 2 * CBLK());
                vflt[3]           = vld1q_f32(converted_filter_c_base + 3 * CBLK());
                vflt[4]           = vld1q_f32(converted_filter_c_base + 4 * CBLK());
                vflt[5]           = vld1q_f32(converted_filter_c_base + 5 * CBLK());
                vflt[6]           = vld1q_f32(converted_filter_c_base + 6 * CBLK());
                vflt[7]           = vld1q_f32(converted_filter_c_base + 7 * CBLK());
                vflt[8]           = vld1q_f32(converted_filter_c_base + 8 * CBLK());
                float32x4_t vbias = vld1q_f32(bias_c_base);

                PRAGMA_OMP_FOR_NOWAIT()
                for (int64_t oh = 0; oh < dst_h; oh++) {
                    const float *input_h_base = input_c_base + oh * 2 * src_w * ICBLK();
                    float *output_h_base      = output_c_base + oh * dst_w * OCBLK();
                    float *sum_h_base         = sum_c_base + oh * dst_w * OCBLK();
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK(), 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK() * 2, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 4, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 4 + ICBLK(), 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 4 + ICBLK() * 2, 0, 3);

                    for (int64_t ow = 0; ow < dst_w_align4; ow += 4) {
                        const float *input_ptr = input_h_base + ow * 2 * ICBLK();
                        float *output_ptr      = output_h_base + ow * OCBLK();
                        __builtin_prefetch(output_ptr, 1, 2);

                        float32x4_t vin[18]; // double buffer
                        float32x4_t vout[4];

                        vout[0] = vbias;
                        vout[1] = vbias;
                        vout[2] = vbias;
                        vout[3] = vbias;

                        vin[0] = vld1q_f32(input_ptr);
                        vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);
                        vin[3] = vld1q_f32(input_ptr + ICBLK() * 3);
                        vin[4] = vld1q_f32(input_ptr + ICBLK() * 4);
                        vin[5] = vld1q_f32(input_ptr + ICBLK() * 5);
                        vin[6] = vld1q_f32(input_ptr + ICBLK() * 6);
                        vin[7] = vld1q_f32(input_ptr + ICBLK() * 7);
                        vin[8] = vld1q_f32(input_ptr + ICBLK() * 8);

                        vin[9]  = vld1q_f32(input_ptr + src_w * ICBLK());
                        vin[10] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                        vin[11] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                        vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 3);
                        vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 4);
                        vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 5);
                        vin[15] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 6);
                        vin[16] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 7);
                        vin[17] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 8);

                        vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                        vout[1] = vfmaq_f32(vout[1], vin[2], vflt[0]);
                        vout[2] = vfmaq_f32(vout[2], vin[4], vflt[0]);
                        vout[3] = vfmaq_f32(vout[3], vin[6], vflt[0]);

                        vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                        vout[1] = vfmaq_f32(vout[1], vin[3], vflt[1]);
                        vout[2] = vfmaq_f32(vout[2], vin[5], vflt[1]);
                        vout[3] = vfmaq_f32(vout[3], vin[7], vflt[1]);

                        vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                        vout[1] = vfmaq_f32(vout[1], vin[4], vflt[2]);
                        vout[2] = vfmaq_f32(vout[2], vin[6], vflt[2]);
                        vout[3] = vfmaq_f32(vout[3], vin[8], vflt[2]);

                        vin[0] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                        vin[1] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                        vin[2] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                        vin[3] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 3);
                        vin[4] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 4);
                        vin[5] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 5);
                        vin[6] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 6);
                        vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 7);
                        vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 8);

                        vout[0] = vfmaq_f32(vout[0], vin[9], vflt[3]);
                        vout[1] = vfmaq_f32(vout[1], vin[11], vflt[3]);
                        vout[2] = vfmaq_f32(vout[2], vin[13], vflt[3]);
                        vout[3] = vfmaq_f32(vout[3], vin[15], vflt[3]);

                        vout[0] = vfmaq_f32(vout[0], vin[10], vflt[4]);
                        vout[1] = vfmaq_f32(vout[1], vin[12], vflt[4]);
                        vout[2] = vfmaq_f32(vout[2], vin[14], vflt[4]);
                        vout[3] = vfmaq_f32(vout[3], vin[16], vflt[4]);

                        vout[0] = vfmaq_f32(vout[0], vin[11], vflt[5]);
                        vout[1] = vfmaq_f32(vout[1], vin[13], vflt[5]);
                        vout[2] = vfmaq_f32(vout[2], vin[15], vflt[5]);
                        vout[3] = vfmaq_f32(vout[3], vin[17], vflt[5]);

                        vout[0] = vfmaq_f32(vout[0], vin[0], vflt[6]);
                        vout[1] = vfmaq_f32(vout[1], vin[2], vflt[6]);
                        vout[2] = vfmaq_f32(vout[2], vin[4], vflt[6]);
                        vout[3] = vfmaq_f32(vout[3], vin[6], vflt[6]);

                        vout[0] = vfmaq_f32(vout[0], vin[1], vflt[7]);
                        vout[1] = vfmaq_f32(vout[1], vin[3], vflt[7]);
                        vout[2] = vfmaq_f32(vout[2], vin[5], vflt[7]);
                        vout[3] = vfmaq_f32(vout[3], vin[7], vflt[7]);

                        vout[0] = vfmaq_f32(vout[0], vin[2], vflt[8]);
                        vout[1] = vfmaq_f32(vout[1], vin[4], vflt[8]);
                        vout[2] = vfmaq_f32(vout[2], vin[6], vflt[8]);
                        vout[3] = vfmaq_f32(vout[3], vin[8], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            float *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                            vout[1]        = vaddq_f32(vout[1], vld1q_f32(sum_ptr + OCBLK() * 1));
                            vout[2]        = vaddq_f32(vout[2], vld1q_f32(sum_ptr + OCBLK() * 2));
                            vout[3]        = vaddq_f32(vout[3], vld1q_f32(sum_ptr + OCBLK() * 3));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            float32x4_t vzero = vdupq_n_f32(0.0f);
                            vout[0]           = vmaxq_f32(vout[0], vzero);
                            vout[1]           = vmaxq_f32(vout[1], vzero);
                            vout[2]           = vmaxq_f32(vout[2], vzero);
                            vout[3]           = vmaxq_f32(vout[3], vzero);
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            float32x4_t vsix = vdupq_n_f32(6.0f);
                            vout[0]          = vminq_f32(vout[0], vsix);
                            vout[1]          = vminq_f32(vout[1], vsix);
                            vout[2]          = vminq_f32(vout[2], vsix);
                            vout[3]          = vminq_f32(vout[3], vsix);
                        }

                        vst1q_f32(output_ptr, vout[0]);
                        vst1q_f32(output_ptr + OCBLK() * 1, vout[1]);
                        vst1q_f32(output_ptr + OCBLK() * 2, vout[2]);
                        vst1q_f32(output_ptr + OCBLK() * 3, vout[3]);
                    }
                    for (int64_t ow = dst_w_align4; ow < dst_w; ow++) {
                        const float *input_ptr = input_h_base + ow * 2 * ICBLK();
                        float *output_ptr      = output_h_base + ow * OCBLK();

                        float32x4_t vin[18];
                        float32x4_t vout[1];

                        vout[0] = vbias;

                        vin[0] = vld1q_f32(input_ptr);
                        vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                        vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);

                        vin[6] = vld1q_f32(input_ptr + src_w * ICBLK());
                        vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                        vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);

                        vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                        vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                        vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);

                        vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                        vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                        vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                        vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                        vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                        vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                        vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                        vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                        vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);

                        if (fuse_flag & conv_fuse_flag::SUM) {
                            float *sum_ptr = sum_h_base + ow * OCBLK();
                            vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU) {
                            vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                        }
                        if (fuse_flag & conv_fuse_flag::RELU6) {
                            vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                        }

                        vst1q_f32(output_ptr, vout[0]);
                    }
                }
#pragma GCC diagnostic pop
            }
        }
    }
}

template <>
void conv_n4cx_depthwise_f3sx_h1w4<1, 2>(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t fltC,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    PRAGMA_OMP_PARALLEL()
    {
        int64_t oh_inner_start, oh_inner_end;
        int64_t ow_inner_start, ow_inner_end;
        oh_inner_start              = 1; // inclusive index
        ow_inner_start              = 1; // inclusive index
        oh_inner_end                = (src_h - 2) / 2 + 1; // exclusive index
        ow_inner_end                = (src_w - 2) / 2 + 1; // exclusive index
        oh_inner_end                = std::max(oh_inner_end, oh_inner_start);
        ow_inner_end                = std::max(ow_inner_end, ow_inner_start);
        int64_t ow_inner_end_align4 = ((ow_inner_end - ow_inner_start) & (~3)) + ow_inner_start;

        const int64_t fltC_pck            = CEIL4(fltC);
        const int64_t src_hW                = src_h * src_w;
        const int64_t dst_hW               = dst_h * dst_w;
        const int64_t input_batch_stride  = fltC_pck * src_hW;
        const int64_t output_batch_stride = fltC_pck * dst_hW;

        int p_c = -1;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        float32x4_t vflt[9];
        float32x4_t vbias;

        PRAGMA_OMP_FOR_COLLAPSE_NOWAIT(3)
        for (int64_t b = 0; b < num_batch; b++) {
            for (int64_t c = 0; c < fltC; c += CBLK()) {
                for (int64_t oh = 0; oh < dst_h; oh++) {
                    const float *converted_filter_c_base = converted_filter + c * 9;
                    const float *bias_c_base             = bias + c;
                    const float *input_c_base            = input + b * input_batch_stride + c * src_hW;
                    float *output_c_base                 = output + b * output_batch_stride + c * dst_hW;
                    float *sum_c_base                    = sum + b * output_batch_stride + c * dst_hW;

                    if (p_c != c) {
                        vflt[0] = vld1q_f32(converted_filter_c_base + 0 * CBLK());
                        vflt[1] = vld1q_f32(converted_filter_c_base + 1 * CBLK());
                        vflt[2] = vld1q_f32(converted_filter_c_base + 2 * CBLK());
                        vflt[3] = vld1q_f32(converted_filter_c_base + 3 * CBLK());
                        vflt[4] = vld1q_f32(converted_filter_c_base + 4 * CBLK());
                        vflt[5] = vld1q_f32(converted_filter_c_base + 5 * CBLK());
                        vflt[6] = vld1q_f32(converted_filter_c_base + 6 * CBLK());
                        vflt[7] = vld1q_f32(converted_filter_c_base + 7 * CBLK());
                        vflt[8] = vld1q_f32(converted_filter_c_base + 8 * CBLK());
                        vbias   = vld1q_f32(bias_c_base);
                        p_c     = c;
                    }

                    const int64_t ih          = -1 + oh * 2;
                    const float *input_h_base = input_c_base + ih * src_w * ICBLK();
                    float *output_h_base      = output_c_base + oh * dst_w * OCBLK();
                    float *sum_h_base         = sum_c_base + oh * dst_w * OCBLK();
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK(), 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 3 + ICBLK() * 2, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 4, 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 4 + ICBLK(), 0, 3);
                    __builtin_prefetch(input_h_base + src_w * ICBLK() * 4 + ICBLK() * 2, 0, 3);

                    if (oh == 0 || oh >= oh_inner_end) {
                        bool ih0_valid = (ih >= 0);
                        bool ih2_valid = (ih + 2 < src_h);

                        {
                            __builtin_prefetch(output_h_base, 1, 2);
                            const float *input_ptr = input_h_base;
                            bool iw2_valid         = (1 < src_w);

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            if (ih0_valid) {
                                vin[1]  = vld1q_f32(input_ptr);
                                vin[2]  = (iw2_valid) ? vld1q_f32(input_ptr + ICBLK()) : vdupq_n_f32(0.0f);
                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            }

                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[8]  = (iw2_valid) ? vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK()) : vdupq_n_f32(0.0f);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);

                            if (ih2_valid) {
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[14] = (iw2_valid) ? vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK()) : vdupq_n_f32(0.0f);
                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                                vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_h_base, vout[0]);
                        }
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow += 4) {
                            const float *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();
                            float *output_ptr      = output_h_base + ow * OCBLK();
                            __builtin_prefetch(output_ptr, 1, 2);

                            float32x4_t vin[18];
                            float32x4_t vout[4];

                            vout[0] = vbias;
                            vout[1] = vbias;
                            vout[2] = vbias;
                            vout[3] = vbias;
                            if (ih0_valid) {
                                vin[0] = vld1q_f32(input_ptr);
                                vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                                vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);
                                vin[3] = vld1q_f32(input_ptr + ICBLK() * 3);
                                vin[4] = vld1q_f32(input_ptr + ICBLK() * 4);
                                vin[5] = vld1q_f32(input_ptr + ICBLK() * 5);
                                vin[6] = vld1q_f32(input_ptr + ICBLK() * 6);
                                vin[7] = vld1q_f32(input_ptr + ICBLK() * 7);
                                vin[8] = vld1q_f32(input_ptr + ICBLK() * 8);

                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                                vout[1] = vfmaq_f32(vout[1], vin[2], vflt[0]);
                                vout[2] = vfmaq_f32(vout[2], vin[4], vflt[0]);
                                vout[3] = vfmaq_f32(vout[3], vin[6], vflt[0]);

                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                                vout[1] = vfmaq_f32(vout[1], vin[3], vflt[1]);
                                vout[2] = vfmaq_f32(vout[2], vin[5], vflt[1]);
                                vout[3] = vfmaq_f32(vout[3], vin[7], vflt[1]);

                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                                vout[1] = vfmaq_f32(vout[1], vin[4], vflt[2]);
                                vout[2] = vfmaq_f32(vout[2], vin[6], vflt[2]);
                                vout[3] = vfmaq_f32(vout[3], vin[8], vflt[2]);
                            }

                            vin[9]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[10] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                            vin[11] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 3);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 4);
                            vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 5);
                            vin[15] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 6);
                            vin[16] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 7);
                            vin[17] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 8);

                            vout[0] = vfmaq_f32(vout[0], vin[9], vflt[3]);
                            vout[1] = vfmaq_f32(vout[1], vin[11], vflt[3]);
                            vout[2] = vfmaq_f32(vout[2], vin[13], vflt[3]);
                            vout[3] = vfmaq_f32(vout[3], vin[15], vflt[3]);

                            vout[0] = vfmaq_f32(vout[0], vin[10], vflt[4]);
                            vout[1] = vfmaq_f32(vout[1], vin[12], vflt[4]);
                            vout[2] = vfmaq_f32(vout[2], vin[14], vflt[4]);
                            vout[3] = vfmaq_f32(vout[3], vin[16], vflt[4]);

                            vout[0] = vfmaq_f32(vout[0], vin[11], vflt[5]);
                            vout[1] = vfmaq_f32(vout[1], vin[13], vflt[5]);
                            vout[2] = vfmaq_f32(vout[2], vin[15], vflt[5]);
                            vout[3] = vfmaq_f32(vout[3], vin[17], vflt[5]);

                            if (ih2_valid) {
                                vin[0] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[1] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                                vin[2] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                                vin[3] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 3);
                                vin[4] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 4);
                                vin[5] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 5);
                                vin[6] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 6);
                                vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 7);
                                vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 8);

                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[6]);
                                vout[1] = vfmaq_f32(vout[1], vin[2], vflt[6]);
                                vout[2] = vfmaq_f32(vout[2], vin[4], vflt[6]);
                                vout[3] = vfmaq_f32(vout[3], vin[6], vflt[6]);

                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[7]);
                                vout[1] = vfmaq_f32(vout[1], vin[3], vflt[7]);
                                vout[2] = vfmaq_f32(vout[2], vin[5], vflt[7]);
                                vout[3] = vfmaq_f32(vout[3], vin[7], vflt[7]);

                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[8]);
                                vout[1] = vfmaq_f32(vout[1], vin[4], vflt[8]);
                                vout[2] = vfmaq_f32(vout[2], vin[6], vflt[8]);
                                vout[3] = vfmaq_f32(vout[3], vin[8], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                                vout[1]        = vaddq_f32(vout[1], vld1q_f32(sum_ptr + OCBLK() * 1));
                                vout[2]        = vaddq_f32(vout[2], vld1q_f32(sum_ptr + OCBLK() * 2));
                                vout[3]        = vaddq_f32(vout[3], vld1q_f32(sum_ptr + OCBLK() * 3));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                float32x4_t vzero = vdupq_n_f32(0.0f);
                                vout[0]           = vmaxq_f32(vout[0], vzero);
                                vout[1]           = vmaxq_f32(vout[1], vzero);
                                vout[2]           = vmaxq_f32(vout[2], vzero);
                                vout[3]           = vmaxq_f32(vout[3], vzero);
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                float32x4_t vsix = vdupq_n_f32(6.0f);
                                vout[0]          = vminq_f32(vout[0], vsix);
                                vout[1]          = vminq_f32(vout[1], vsix);
                                vout[2]          = vminq_f32(vout[2], vsix);
                                vout[3]          = vminq_f32(vout[3], vsix);
                            }

                            vst1q_f32(output_ptr, vout[0]);
                            vst1q_f32(output_ptr + OCBLK(), vout[1]);
                            vst1q_f32(output_ptr + OCBLK() * 2, vout[2]);
                            vst1q_f32(output_ptr + OCBLK() * 3, vout[3]);
                        }
                        for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                            __builtin_prefetch(output_h_base + ow * OCBLK(), 1, 2);
                            const float *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;
                            if (ih0_valid) {
                                vin[0]  = vld1q_f32(input_ptr);
                                vin[1]  = vld1q_f32(input_ptr + ICBLK());
                                vin[2]  = vld1q_f32(input_ptr + ICBLK() * 2);
                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                                vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            }

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());
                            vin[8]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);

                            if (ih2_valid) {
                                vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());
                                vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                                vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                                vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_h_base + ow * OCBLK(), vout[0]);
                        }
                        if (ow_inner_end < dst_w) { // NOTE: when in_size, k_size and stride are unmatched, the tail is no more than 1.
                            const float *input_ptr = input_h_base + (ow_inner_end * 2 - 1) * ICBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;
                            if (ih0_valid) {
                                vin[0]  = vld1q_f32(input_ptr);
                                vin[1]  = vld1q_f32(input_ptr + ICBLK());
                                vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                                vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            }

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);

                            if (ih2_valid) {
                                vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                                vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());
                                vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                                vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                            }

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base + ow_inner_end * OCBLK()));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_c_base + oh * dst_w * OCBLK() + ow_inner_end * OCBLK(), vout[0]);
                        }
                    } else {
                        {
                            __builtin_prefetch(output_h_base, 1, 2);
                            const float *input_ptr = input_h_base;
                            bool iw2_valid         = (1 < src_w);

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            vin[1]  = vld1q_f32(input_ptr);
                            vin[2]  = (iw2_valid) ? vld1q_f32(input_ptr + ICBLK()) : vdupq_n_f32(0.0f);
                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);

                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[8]  = (iw2_valid) ? vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK()) : vdupq_n_f32(0.0f);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);

                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[14] = (iw2_valid) ? vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK()) : vdupq_n_f32(0.0f);
                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_h_base, vout[0]);
                        }
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end_align4; ow += 4) {
                            const float *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();
                            float *output_ptr      = output_h_base + ow * OCBLK();
                            __builtin_prefetch(output_ptr, 1, 2);

                            float32x4_t vin[18];
                            float32x4_t vout[4];

                            vout[0] = vbias;
                            vout[1] = vbias;
                            vout[2] = vbias;
                            vout[3] = vbias;

                            vin[0] = vld1q_f32(input_ptr);
                            vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                            vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);
                            vin[3] = vld1q_f32(input_ptr + ICBLK() * 3);
                            vin[4] = vld1q_f32(input_ptr + ICBLK() * 4);
                            vin[5] = vld1q_f32(input_ptr + ICBLK() * 5);
                            vin[6] = vld1q_f32(input_ptr + ICBLK() * 6);
                            vin[7] = vld1q_f32(input_ptr + ICBLK() * 7);
                            vin[8] = vld1q_f32(input_ptr + ICBLK() * 8);

                            vin[9]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[10] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                            vin[11] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);
                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 3);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 4);
                            vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 5);
                            vin[15] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 6);
                            vin[16] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 7);
                            vin[17] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 8);

                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                            vout[1] = vfmaq_f32(vout[1], vin[2], vflt[0]);
                            vout[2] = vfmaq_f32(vout[2], vin[4], vflt[0]);
                            vout[3] = vfmaq_f32(vout[3], vin[6], vflt[0]);

                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[1] = vfmaq_f32(vout[1], vin[3], vflt[1]);
                            vout[2] = vfmaq_f32(vout[2], vin[5], vflt[1]);
                            vout[3] = vfmaq_f32(vout[3], vin[7], vflt[1]);

                            vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            vout[1] = vfmaq_f32(vout[1], vin[4], vflt[2]);
                            vout[2] = vfmaq_f32(vout[2], vin[6], vflt[2]);
                            vout[3] = vfmaq_f32(vout[3], vin[8], vflt[2]);

                            vin[0] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[1] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                            vin[2] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);
                            vin[3] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 3);
                            vin[4] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 4);
                            vin[5] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 5);
                            vin[6] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 6);
                            vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 7);
                            vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 8);

                            vout[0] = vfmaq_f32(vout[0], vin[9], vflt[3]);
                            vout[1] = vfmaq_f32(vout[1], vin[11], vflt[3]);
                            vout[2] = vfmaq_f32(vout[2], vin[13], vflt[3]);
                            vout[3] = vfmaq_f32(vout[3], vin[15], vflt[3]);

                            vout[0] = vfmaq_f32(vout[0], vin[10], vflt[4]);
                            vout[1] = vfmaq_f32(vout[1], vin[12], vflt[4]);
                            vout[2] = vfmaq_f32(vout[2], vin[14], vflt[4]);
                            vout[3] = vfmaq_f32(vout[3], vin[16], vflt[4]);

                            vout[0] = vfmaq_f32(vout[0], vin[11], vflt[5]);
                            vout[1] = vfmaq_f32(vout[1], vin[13], vflt[5]);
                            vout[2] = vfmaq_f32(vout[2], vin[15], vflt[5]);
                            vout[3] = vfmaq_f32(vout[3], vin[17], vflt[5]);

                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[6]);
                            vout[1] = vfmaq_f32(vout[1], vin[2], vflt[6]);
                            vout[2] = vfmaq_f32(vout[2], vin[4], vflt[6]);
                            vout[3] = vfmaq_f32(vout[3], vin[6], vflt[6]);

                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[7]);
                            vout[1] = vfmaq_f32(vout[1], vin[3], vflt[7]);
                            vout[2] = vfmaq_f32(vout[2], vin[5], vflt[7]);
                            vout[3] = vfmaq_f32(vout[3], vin[7], vflt[7]);

                            vout[0] = vfmaq_f32(vout[0], vin[2], vflt[8]);
                            vout[1] = vfmaq_f32(vout[1], vin[4], vflt[8]);
                            vout[2] = vfmaq_f32(vout[2], vin[6], vflt[8]);
                            vout[3] = vfmaq_f32(vout[3], vin[8], vflt[8]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                                vout[1]        = vaddq_f32(vout[1], vld1q_f32(sum_ptr + OCBLK() * 1));
                                vout[2]        = vaddq_f32(vout[2], vld1q_f32(sum_ptr + OCBLK() * 2));
                                vout[3]        = vaddq_f32(vout[3], vld1q_f32(sum_ptr + OCBLK() * 3));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                float32x4_t vzero = vdupq_n_f32(0.0f);
                                vout[0]           = vmaxq_f32(vout[0], vzero);
                                vout[1]           = vmaxq_f32(vout[1], vzero);
                                vout[2]           = vmaxq_f32(vout[2], vzero);
                                vout[3]           = vmaxq_f32(vout[3], vzero);
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                float32x4_t vsix = vdupq_n_f32(6.0f);
                                vout[0]          = vminq_f32(vout[0], vsix);
                                vout[1]          = vminq_f32(vout[1], vsix);
                                vout[2]          = vminq_f32(vout[2], vsix);
                                vout[3]          = vminq_f32(vout[3], vsix);
                            }

                            vst1q_f32(output_ptr, vout[0]);
                            vst1q_f32(output_ptr + OCBLK() * 1, vout[1]);
                            vst1q_f32(output_ptr + OCBLK() * 2, vout[2]);
                            vst1q_f32(output_ptr + OCBLK() * 3, vout[3]);
                        }
                        for (int64_t ow = ow_inner_end_align4; ow < ow_inner_end; ow++) {
                            const float *input_ptr = input_h_base + (ow * 2 - 1) * ICBLK();
                            float *output_ptr      = output_h_base + ow * OCBLK();
                            __builtin_prefetch(output_ptr, 1, 2);

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            vin[0] = vld1q_f32(input_ptr);
                            vin[1] = vld1q_f32(input_ptr + ICBLK() * 1);
                            vin[2] = vld1q_f32(input_ptr + ICBLK() * 2);

                            vin[6] = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 1);
                            vin[8] = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK() * 2);

                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 1);
                            vin[14] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK() * 2);

                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);
                            vout[0] = vfmaq_f32(vout[0], vin[2], vflt[2]);
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);
                            vout[0] = vfmaq_f32(vout[0], vin[8], vflt[5]);
                            vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);
                            vout[0] = vfmaq_f32(vout[0], vin[14], vflt[8]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                float *sum_ptr = sum_h_base + ow * OCBLK();
                                vout[0]        = vaddq_f32(vout[0], vld1q_f32(sum_ptr));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_ptr, vout[0]);
                        }
                        if (ow_inner_end < dst_w) { // NOTE: when in_size, k_size and stride are unmatched, the tail is no more than 1.
                            const float *input_ptr = input_h_base + (ow_inner_end * 2 - 1) * ICBLK();

                            float32x4_t vin[18];
                            float32x4_t vout[1];

                            vout[0] = vbias;

                            vin[0]  = vld1q_f32(input_ptr);
                            vin[1]  = vld1q_f32(input_ptr + ICBLK());
                            vout[0] = vfmaq_f32(vout[0], vin[0], vflt[0]);
                            vout[0] = vfmaq_f32(vout[0], vin[1], vflt[1]);

                            vin[6]  = vld1q_f32(input_ptr + src_w * ICBLK());
                            vin[7]  = vld1q_f32(input_ptr + src_w * ICBLK() + ICBLK());
                            vout[0] = vfmaq_f32(vout[0], vin[6], vflt[3]);
                            vout[0] = vfmaq_f32(vout[0], vin[7], vflt[4]);

                            vin[12] = vld1q_f32(input_ptr + src_w * ICBLK() * 2);
                            vin[13] = vld1q_f32(input_ptr + src_w * ICBLK() * 2 + ICBLK());
                            vout[0] = vfmaq_f32(vout[0], vin[12], vflt[6]);
                            vout[0] = vfmaq_f32(vout[0], vin[13], vflt[7]);

                            if (fuse_flag & conv_fuse_flag::SUM) {
                                vout[0] = vaddq_f32(vout[0], vld1q_f32(sum_h_base + ow_inner_end * OCBLK()));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU) {
                                vout[0] = vmaxq_f32(vout[0], vdupq_n_f32(0.0f));
                            }
                            if (fuse_flag & conv_fuse_flag::RELU6) {
                                vout[0] = vminq_f32(vout[0], vdupq_n_f32(6.0f));
                            }

                            vst1q_f32(output_c_base + oh * dst_w * OCBLK() + ow_inner_end * OCBLK(), vout[0]);
                        }
                    }
                }
            }
        }
#pragma GCC diagnostic pop
    }
}

void conv_n4cx_depthwise_f3sx_convolution(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t fltC,
    const int64_t padding,
    const int64_t stride,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    int64_t case_id = padding * 10 + stride;

    switch (case_id) {
        case 01: // p0s1
            conv_n4cx_depthwise_f3sx_h1w4<0, 1>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                src_h,
                src_w,
                dst_h,
                dst_w,
                num_batch,
                fuse_flag);
            return;

        case 11: // p1s1
            conv_n4cx_depthwise_f3sx_h1w4<1, 1>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                src_h,
                src_w,
                dst_h,
                dst_w,
                num_batch,
                fuse_flag);
            return;

        case 02: // p0s2
            conv_n4cx_depthwise_f3sx_h1w4<0, 2>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                src_h,
                src_w,
                dst_h,
                dst_w,
                num_batch,
                fuse_flag);
            return;

        case 12: // p1s2
            conv_n4cx_depthwise_f3sx_h1w4<1, 2>(
                converted_filter,
                bias,
                input,
                output,
                sum,
                fltC,
                src_h,
                src_w,
                dst_h,
                dst_w,
                num_batch,
                fuse_flag);
            return;

        default:
            return;
    }
}

void conv_n4cx_depthwise_general_convolution(
    const float *converted_filter,
    const float *bias,
    const float *input,
    float *output,
    float *sum,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t fltC,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t strd_h,
    const int64_t strd_w,
    const int64_t dltn_h,
    const int64_t dltn_w,
    const int64_t num_batch,
    const uint32_t fuse_flag)
{
    int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((pad_w - 0 * dltn_w), strd_w)); // inclusive
    int64_t ow_inner_end   = std::min((int64_t)dst_w, DIV_CEIL((src_w + pad_w - (flt_w - 1) * dltn_w), strd_w)); // exclusive
    ow_inner_start         = std::min(ow_inner_start, dst_w);
    ow_inner_end           = std::max(ow_inner_end, ow_inner_start);

    constexpr int otw          = 8;
    int64_t ow_inner_end_align = ((ow_inner_end - ow_inner_start) / otw * otw) + ow_inner_start;

    const int64_t fltC_pck            = CEIL4(fltC);
    const int64_t src_hW                = src_h * src_w;
    const int64_t dst_hW               = dst_h * dst_w;
    const int64_t input_batch_stride  = fltC_pck * src_hW;
    const int64_t output_batch_stride = fltC_pck * dst_hW;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t b = 0; b < num_batch; b++) {
        for (int64_t c = 0; c < fltC_pck; c += CBLK()) {
            for (int64_t oh = 0; oh < dst_h; oh++) {
                const float *cvt_filter_c_base = converted_filter + c * flt_h * flt_w;
                float32x4_t vbias              = vld1q_f32(bias + c);
                const float *input_c_base      = input + b * input_batch_stride + c * src_hW;
                float *output_c_base           = output + b * output_batch_stride + c * dst_hW;
                float *sum_c_base              = sum + b * output_batch_stride + c * dst_hW;

                const int64_t ih_base = -pad_h + oh * strd_h;

                const int64_t flt_h_start = std::max(-ih_base + dltn_h - 1, (int64_t)0) / dltn_h; // inclusive
                const int64_t flt_h_end   = std::min(flt_h, (src_h - ih_base + dltn_h - 1) / dltn_h); // exclusive
                if (flt_h_end - flt_h_start <= 0) continue;

                const float *input_h_base = input_c_base + ih_base * src_w * ICBLK();
                float *output_h_base      = output_c_base + oh * dst_w * OCBLK();
                float *sum_h_base         = sum_c_base + oh * dst_w * OCBLK();

                for (int64_t ow = 0; ow < ow_inner_start; ow++) {
                    int64_t iw_base    = -pad_w + ow * strd_w;
                    int64_t flt_w_start = std::max(-iw_base + dltn_w - 1, (int64_t)0) / dltn_w; // inclusive
                    int64_t flt_w_end   = std::min(flt_w, (src_w - iw_base + dltn_w - 1) / dltn_w); // exclusive

                    conv_n4cx_depthwise_general_h1w1_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(), vbias, src_w, flt_w, ih_base, iw_base, flt_h_start, flt_h_end, flt_w_start, flt_w_end, dltn_h, dltn_w, fuse_flag);
                }
                for (int64_t ow = ow_inner_start; ow < ow_inner_end_align; ow += otw) {
                    int64_t iw_base = -pad_w + ow * strd_w;

                    conv_n4cx_depthwise_general_h1w8_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(), vbias, flt_w, strd_w, ih_base, iw_base, flt_h_start, flt_h_end, dltn_h * src_w, dltn_w, fuse_flag);
                }
                for (int64_t ow = ow_inner_end_align; ow < ow_inner_end; ow++) {
                    int64_t iw_base = -pad_w + ow * strd_w;

                    conv_n4cx_depthwise_general_h1w1_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(), vbias, src_w, flt_w, ih_base, iw_base, flt_h_start, flt_h_end, 0, flt_w, dltn_h, dltn_w, fuse_flag);
                }
                for (int64_t ow = ow_inner_end; ow < dst_w; ow++) {
                    int64_t iw_base    = -pad_w + ow * strd_w;
                    int64_t flt_w_start = std::max(-iw_base + dltn_w - 1, (int64_t)0) / dltn_w; // inclusive
                    int64_t flt_w_end   = std::min(flt_w, (src_w - iw_base + dltn_w - 1) / dltn_w); // exclusive

                    conv_n4cx_depthwise_general_h1w1_kernel(
                        cvt_filter_c_base, input_h_base + iw_base * ICBLK(), output_h_base + ow * OCBLK(), sum_h_base + ow * OCBLK(), vbias, src_w, flt_w, ih_base, iw_base, flt_h_start, flt_h_end, flt_w_start, flt_w_end, dltn_h, dltn_w, fuse_flag);
                }
            }
        }
    }
}

uint64_t conv2d_n4cx_depthwise_fp32_runtime_executor::cal_temp_buffer_size()
{
    return 0;
}

void conv2d_n4cx_depthwise_fp32_runtime_executor::adjust_schedule_param()
{
    return;
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_runtime_executor::execute()
{
    const conv2d_param &cp = *conv_param_;

    const float *input            = (const float *)src_;
    const float *converted_filter = (const float *)cvt_filter_;
    const float *bias             = (const float *)cvt_bias_;
    float *output                 = (float *)dst_;
    float *sum                    = (float *)sum_;

    const int64_t src_h       = src_shape_->GetDim(2);
    const int64_t src_w       = src_shape_->GetDim(3);
    const int64_t num_output      = cp.num_output;
    const int64_t dst_h      = dst_shape_->GetDim(2);
    const int64_t dst_w      = dst_shape_->GetDim(3);
    const int64_t flt_h      = cp.kernel_h;
    const int64_t flt_w      = cp.kernel_w;
    const int64_t pad_h      = cp.pad_h;
    const int64_t pad_w      = cp.pad_w;
    const int64_t strd_h     = cp.stride_h;
    const int64_t strd_w     = cp.stride_w;
    const int64_t dltn_h     = cp.dilation_h;
    const int64_t dltn_w     = cp.dilation_w;
    const int64_t num_batch = src_shape_->GetDim(0);

    if (flt_h == 3 && flt_w == 3 &&
        pad_h < 2 && pad_h == pad_w &&
        strd_h < 3 && strd_h == strd_w &&
        dltn_h == 1 && dltn_w == 1) {
        conv_n4cx_depthwise_f3sx_convolution(
            converted_filter,
            bias,
            input,
            output,
            sum,
            src_h,
            src_w,
            dst_h,
            dst_w,
            num_output,
            pad_h,
            strd_h,
            num_batch,
            cp.fuse_flag);
    } else {
        conv_n4cx_depthwise_general_convolution(
            converted_filter,
            bias,
            input,
            output,
            sum,
            src_h,
            src_w,
            dst_h,
            dst_w,
            num_output,
            flt_h,
            flt_w,
            pad_h,
            pad_w,
            strd_h,
            strd_w,
            dltn_h,
            dltn_w,
            num_batch,
            cp.fuse_flag);
    }
    return ppl::common::RC_SUCCESS;
}

size_t conv_n4cx_depthwise_get_converted_filter_size(
    const int64_t num_output,
    const int64_t flt_h,
    const int64_t flt_w)
{
    return CEIL128(CEIL4(num_output) * flt_h * flt_w * sizeof(float));
}

void conv_n4cx_depthwise_convert_filter(
    const float *filter,
    float *converted_filter,
    const int64_t num_output,
    const int64_t flt_h,
    const int64_t flt_w)
{
    const int64_t oc_floor4 = FLOOR4(num_output);
    const int64_t flt_hw    = flt_h * flt_w;

    for (int64_t c = 0; c < oc_floor4; c += CBLK()) {
        for (int64_t idx = 0; idx < flt_hw; idx++) {
            for (int64_t c_in = 0; c_in < CBLK(); c_in++) {
                converted_filter[flt_hw * c + idx * CBLK() + c_in] = filter[flt_hw * c + c_in * flt_hw + idx];
            }
        }
    }

    const int64_t oc_tail = num_output - oc_floor4;
    if (oc_tail) {
        for (int64_t idx = 0; idx < flt_hw; idx++) {
            for (int64_t c = 0; c < oc_tail; c++) {
                converted_filter[flt_hw * oc_floor4 + idx * CBLK() + c] = filter[flt_hw * oc_floor4 + c * flt_hw + idx];
            }
            for (int64_t c = oc_tail; c < CBLK(); c++) {
                converted_filter[flt_hw * oc_floor4 + idx * CBLK() + c] = 0.0f;
            }
        }
    }
}

bool conv2d_n4cx_depthwise_fp32_offline_manager::is_supported()
{
    return (param_.group == param_.channels) && (param_.group == param_.num_output);
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_offline_manager::fast_init_schedule_param()
{
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_offline_manager::pick_best_schedule_param(
    const ppl::common::TensorShape &src_shape,
    void *src,
    void *cvt_bias,
    const ppl::common::TensorShape &dst_shape,
    void *dst,
    bool tune_sp,
    double &run_time)
{
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_offline_manager::try_fuse(conv_fuse_flag_t fuse_type)
{
    return ((fuse_type | conv_fuse_flag::HSWISH) || (fuse_type | conv_fuse_flag::PRELU )) ?
        ppl::common::RC_UNSUPPORTED : ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_offline_manager::generate_cvt_weights_shapes(
    ppl::common::TensorShape &cvt_filter_shape,
    ppl::common::TensorShape &cvt_bias_shape)
{
    const int64_t num_output = param_.num_output;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    cvt_bias_size_ = CEIL4(num_output) * sizeof(float);
    cvt_bias_shape.SetDimCount(1);
    cvt_bias_shape.SetDim(0, cvt_bias_size_/sizeof(float));
    cvt_bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);

    cvt_filter_size_ = conv_n4cx_depthwise_get_converted_filter_size(num_output, kernel_h, kernel_w);
    cvt_filter_shape.SetDimCount(1);
    cvt_filter_shape.SetDim(0, cvt_filter_size_/sizeof(float));
    cvt_filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_depthwise_fp32_offline_manager::generate_cvt_weights(
    const void *filter,
    const void *bias,
    void* new_filter,
    void* new_bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    cvt_bias_size_ = CEIL4(num_output) * sizeof(float);
    if (!bias && new_bias) {
        cvt_bias_ = new_bias;
    } else if (bias && new_bias) {
        cvt_bias_ = new_bias;
        int64_t padding_offset_bytes = num_output * sizeof(float);
        int64_t padding_bytes        = (CEIL4(num_output) - num_output) * sizeof(float);
        std::memcpy(cvt_bias_, bias, num_output * sizeof(float));
        std::memset((uint8_t *)cvt_bias_ + padding_offset_bytes, 0, padding_bytes);
    } else {
        cvt_bias_ = allocator_->Alloc(cvt_bias_size_);
        std::memset(cvt_bias_, 0, cvt_bias_size_);
        is_bias_owner_ = true;
    }

    cvt_filter_size_ = conv_n4cx_depthwise_get_converted_filter_size(num_output, kernel_h, kernel_w);
    cvt_filter_ = new_filter;
    conv_n4cx_depthwise_convert_filter(
        (const float *)filter, (float *)cvt_filter_, num_output, kernel_h, kernel_w);
    
    return ppl::common::RC_SUCCESS;
}

conv2d_runtime_executor *conv2d_n4cx_depthwise_fp32_offline_manager::gen_executor()
{
    return new conv2d_n4cx_depthwise_fp32_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

#undef CBLK
#undef ICBLK
#undef OCBLK

}}}}; // namespace ppl::kernel::arm_server::neon
