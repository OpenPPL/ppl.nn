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

#include "ppl/kernel/arm_server/avepool/neon/avepool.h"

#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

#define CVL() 4

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

static void avepool2d_n4cx_exclude_fp32(
    const float *input,
    float *output,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_channel,
    const int64_t num_batch,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t strd_h,
    const int64_t strd_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t dltn_h,
    const int64_t dltn_w)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        const float32x4_t vzero         = vdupq_n_f32(0.0f);
        for (int64_t n = 0; n < num_batch; n++) {
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel_ceil4; c += CVL()) {
                const float *input_b_base = input + n * num_channel_ceil4 * src_h * src_w;
                float *output_b_base      = output + n * num_channel_ceil4 * dst_h * dst_w;
                const float *input_c_base = input_b_base + c * src_h * src_w;
                float *output_c_base      = output_b_base + c * dst_h * dst_w;
                for (int64_t oh = 0; oh < dst_h; oh++) {
                    const int64_t ih_base     = -pad_h + oh * strd_h;
                    const float *input_h_base = input_c_base + ih_base * src_w * CVL();
                    const int64_t flt_h_start  = std::max(-ih_base + dltn_h - 1, (int64_t)0) / dltn_h; // inclusive
                    const int64_t flt_h_end    = std::min(flt_h, (src_h - ih_base + dltn_h - 1) / dltn_h); // exclusive

                    for (int64_t ow = 0; ow < dst_w; ow++) {
                        int64_t iw_base           = -pad_w + ow * strd_w;
                        const float *input_w_base = input_h_base + iw_base * CVL();
                        int64_t flt_w_start        = std::max(-iw_base + dltn_w - 1, (int64_t)0) / dltn_w; // inclusive
                        int64_t flt_w_end          = std::min(flt_w, (src_w - iw_base + dltn_w - 1) / dltn_w); // exclusive

                        float32x4_t vout       = vzero;
                        float ave_coeff        = 1.0f / ((float)(flt_h_end - flt_h_start) * (flt_w_end - flt_w_start));
                        float32x4_t vave_coeff = vdupq_n_f32(ave_coeff);

                        for (int64_t kh = flt_h_start; kh < flt_h_end; kh++) {
                            for (int64_t kw = flt_w_start; kw < flt_w_end; kw++) {
                                float32x4_t vin = vld1q_f32(input_w_base + (kh * dltn_h * src_w + kw * dltn_w) * CVL());
                                vout            = vaddq_f32(vout, vin);
                            }
                        }
                        vout = vmulq_f32(vout, vave_coeff);

                        vst1q_f32(output_c_base + oh * dst_w * CVL() + ow * CVL(), vout);
                    }
                }
            }
        }
    }
}

static void avepool2d_n4cx_include_fp32(
    const float *input,
    float *output,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t num_channel,
    const int64_t num_batch,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t strd_h,
    const int64_t strd_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t dltn_h,
    const int64_t dltn_w)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        const float32x4_t vzero         = vdupq_n_f32(0.0f);
        const float kernel_size_recp    = 1.0f / (flt_h * flt_w);
        const float32x4_t vave_coeff    = vdupq_n_f32(kernel_size_recp);
        for (int64_t n = 0; n < num_batch; n++) {
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel_ceil4; c += CVL()) {
                const float *input_b_base = input + n * num_channel_ceil4 * src_h * src_w;
                float *output_b_base      = output + n * num_channel_ceil4 * dst_h * dst_w;
                const float *input_c_base = input_b_base + c * src_h * src_w;
                float *output_c_base      = output_b_base + c * dst_h * dst_w;
                for (int64_t oh = 0; oh < dst_h; oh++) {
                    const int64_t ih_base     = -pad_h + oh * strd_h;
                    const float *input_h_base = input_c_base + ih_base * src_w * CVL();
                    const int64_t flt_h_start  = std::max(-ih_base + dltn_h - 1, (int64_t)0) / dltn_h; // inclusive
                    const int64_t flt_h_end    = std::min(flt_h, (src_h - ih_base + dltn_h - 1) / dltn_h); // exclusive

                    for (int64_t ow = 0; ow < dst_w; ow++) {
                        int64_t iw_base           = -pad_w + ow * strd_w;
                        const float *input_w_base = input_h_base + iw_base * CVL();
                        int64_t flt_w_start        = std::max(-iw_base + dltn_w - 1, (int64_t)0) / dltn_w; // inclusive
                        int64_t flt_w_end          = std::min(flt_w, (src_w - iw_base + dltn_w - 1) / dltn_w); // exclusive

                        float32x4_t vout = vzero;

                        for (int64_t kh = flt_h_start; kh < flt_h_end; kh++) {
                            for (int64_t kw = flt_w_start; kw < flt_w_end; kw++) {
                                float32x4_t vin = vld1q_f32(input_w_base + (kh * dltn_h * src_w + kw * dltn_w) * CVL());
                                vout            = vaddq_f32(vout, vin);
                            }
                        }
                        vout = vmulq_f32(vout, vave_coeff);

                        vst1q_f32(output_c_base + oh * dst_w * CVL() + ow * CVL(), vout);
                    }
                }
            }
        }
    }
}

static void avepool2d_n4cx_global_fp32(
    const float *input,
    float *output,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t num_channel,
    const int64_t num_batch)
{
    PRAGMA_OMP_PARALLEL()
    {
        const int64_t num_channel_ceil4 = CEIL4(num_channel);
        const float32x4_t vzero         = vdupq_n_f32(0.0f);
        const float in_size_recp        = 1.0f / (src_h * src_w);
        float32x4_t vave_coeff          = vdupq_n_f32(in_size_recp);

        for (int64_t n = 0; n < num_batch; n++) {
            const float *input_b_base = input + n * num_channel_ceil4 * src_h * src_w;
            float *output_b_base      = output + n * num_channel_ceil4;
            PRAGMA_OMP_FOR_NOWAIT()
            for (int64_t c = 0; c < num_channel_ceil4; c += CVL()) {
                const float *input_c_base = input_b_base + c * src_h * src_w;

                float32x4_t vout = vzero;
                for (int64_t idx = 0; idx < src_h * src_w; idx++) {
                    float32x4_t vin = vld1q_f32(input_c_base + idx * CVL());
                    vout            = vaddq_f32(vout, vin);
                }
                vout = vmulq_f32(vout, vave_coeff);

                vst1q_f32(output_b_base + c, vout);
            }
        }
    }
}

ppl::common::RetCode avepool2d_n4cx_fp32(
    const float *src,
    float *dst,
    const int32_t src_n,
    const int32_t src_c,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    const int32_t global_pooling,
    const bool count_include_pad)
{
    if (global_pooling) {
        avepool2d_n4cx_global_fp32(src, dst, src_h, src_w, src_c, src_n);
    } else if (count_include_pad) {
        avepool2d_n4cx_include_fp32(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    } else {
        avepool2d_n4cx_exclude_fp32(src, dst, src_h, src_w, dst_h, dst_w, src_c, src_n, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    }

    return ppl::common::RC_SUCCESS;
}

}}}}; // namespace ppl::kernel::arm_server::neon
