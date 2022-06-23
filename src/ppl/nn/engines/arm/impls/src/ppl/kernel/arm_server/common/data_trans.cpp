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
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/threading_tools.h"

namespace ppl { namespace kernel { namespace arm_server {

#define TRANSPOSE_8X8(input, output)                                                                                      \
    {                                                                                                                     \
        float16x8x2_t temp_fp16[4];                                                                                       \
        temp_fp16[0] = vtrnq_f16(input[0], input[1]);                                                                     \
        temp_fp16[1] = vtrnq_f16(input[2], input[3]);                                                                     \
        temp_fp16[2] = vtrnq_f16(input[4], input[5]);                                                                     \
        temp_fp16[3] = vtrnq_f16(input[6], input[7]);                                                                     \
        float32x4x2_t temp_fp32[4];                                                                                       \
        temp_fp32[0] = vtrnq_f32((float32x4_t)temp_fp16[0].val[0], (float32x4_t)temp_fp16[1].val[0]);                     \
        temp_fp32[1] = vtrnq_f32((float32x4_t)temp_fp16[0].val[1], (float32x4_t)temp_fp16[1].val[1]);                     \
        temp_fp32[2] = vtrnq_f32((float32x4_t)temp_fp16[2].val[0], (float32x4_t)temp_fp16[3].val[0]);                     \
        temp_fp32[3] = vtrnq_f32((float32x4_t)temp_fp16[2].val[1], (float32x4_t)temp_fp16[3].val[1]);                     \
        output[0]    = (float16x8_t)vcombine_f32(vget_low_f32(temp_fp32[0].val[0]), vget_low_f32(temp_fp32[2].val[0]));   \
        output[1]    = (float16x8_t)vcombine_f32(vget_low_f32(temp_fp32[1].val[0]), vget_low_f32(temp_fp32[3].val[0]));   \
        output[2]    = (float16x8_t)vcombine_f32(vget_low_f32(temp_fp32[0].val[1]), vget_low_f32(temp_fp32[2].val[1]));   \
        output[3]    = (float16x8_t)vcombine_f32(vget_low_f32(temp_fp32[1].val[1]), vget_low_f32(temp_fp32[3].val[1]));   \
        output[4]    = (float16x8_t)vcombine_f32(vget_high_f32(temp_fp32[0].val[0]), vget_high_f32(temp_fp32[2].val[0])); \
        output[5]    = (float16x8_t)vcombine_f32(vget_high_f32(temp_fp32[1].val[0]), vget_high_f32(temp_fp32[3].val[0])); \
        output[6]    = (float16x8_t)vcombine_f32(vget_high_f32(temp_fp32[0].val[1]), vget_high_f32(temp_fp32[2].val[1])); \
        output[7]    = (float16x8_t)vcombine_f32(vget_high_f32(temp_fp32[1].val[1]), vget_high_f32(temp_fp32[3].val[1])); \
    }

ppl::common::RetCode N4cxToNdarrayFp32(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst)
{
    const int64_t c_blk      = 4;
    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c; c += c_blk) {
                const int64_t c_eff = min(channels - c, c_blk);
                const float* p_src  = src + (n * pad_c + c) * inner_dims;
                float* p_dst[c_blk] = {
                    dst + (n * channels + c + 0) * inner_dims,
                    dst + (n * channels + c + 1) * inner_dims,
                    dst + (n * channels + c + 2) * inner_dims,
                    dst + (n * channels + c + 3) * inner_dims};
                if (c_eff == c_blk) {
                    for (int64_t k = 0; k < inner_dims; ++k) {
                        p_dst[0][k] = p_src[k * c_blk + 0];
                        p_dst[1][k] = p_src[k * c_blk + 1];
                        p_dst[2][k] = p_src[k * c_blk + 2];
                        p_dst[3][k] = p_src[k * c_blk + 3];
                    }
                } else {
                    for (int64_t k = 0; k < inner_dims; ++k) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            p_dst[cc][k] = p_src[k * c_blk + cc];
                        }
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(inner_dims, num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c; c += c_blk) {
                    const int64_t c_eff = min(channels - c, c_blk);
                    const float* p_src  = src + (n * pad_c + c) * inner_dims;
                    float* p_dst[c_blk] = {
                        dst + (n * channels + c + 0) * inner_dims,
                        dst + (n * channels + c + 1) * inner_dims,
                        dst + (n * channels + c + 2) * inner_dims,
                        dst + (n * channels + c + 3) * inner_dims};
                    if (c_eff == c_blk) {
                        for (int64_t k = inner_dims_start; k < inner_dims_end; ++k) {
                            p_dst[0][k] = p_src[k * c_blk + 0];
                            p_dst[1][k] = p_src[k * c_blk + 1];
                            p_dst[2][k] = p_src[k * c_blk + 2];
                            p_dst[3][k] = p_src[k * c_blk + 3];
                        }
                    } else {
                        for (int64_t k = inner_dims_start; k < inner_dims_end; ++k) {
                            for (int64_t cc = 0; cc < c_eff; cc++) {
                                p_dst[cc][k] = p_src[k * c_blk + cc];
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode NdarrayToN4cxFp32(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst)
{
    const int64_t c_blk      = 4;
    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c; c += c_blk) {
                const int64_t c_eff       = min(channels - c, c_blk);
                const float* p_src[c_blk] = {
                    src + (n * channels + c + 0) * inner_dims,
                    src + (n * channels + c + 1) * inner_dims,
                    src + (n * channels + c + 2) * inner_dims,
                    src + (n * channels + c + 3) * inner_dims};
                float* p_dst = dst + (n * pad_c + c) * inner_dims;
                if (c_eff == c_blk) {
                    for (int64_t k = 0; k < inner_dims; ++k) {
                        p_dst[k * c_blk + 0] = p_src[0][k];
                        p_dst[k * c_blk + 1] = p_src[1][k];
                        p_dst[k * c_blk + 2] = p_src[2][k];
                        p_dst[k * c_blk + 3] = p_src[3][k];
                    }
                } else {
                    for (int64_t k = 0; k < inner_dims; ++k) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            p_dst[k * c_blk + cc] = p_src[cc][k];
                        }
                        for (int64_t cc = c_eff; cc < c_blk; cc++) { // pad channels to zero
                            p_dst[k * c_blk + cc] = 0;
                        }
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(inner_dims, num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c; c += c_blk) {
                    const int64_t c_eff       = min(channels - c, c_blk);
                    const float* p_src[c_blk] = {
                        src + (n * channels + c + 0) * inner_dims,
                        src + (n * channels + c + 1) * inner_dims,
                        src + (n * channels + c + 2) * inner_dims,
                        src + (n * channels + c + 3) * inner_dims};
                    float* p_dst = dst + (n * pad_c + c) * inner_dims;
                    if (c_eff == c_blk) {
                        for (int64_t k = inner_dims_start; k < inner_dims_end; ++k) {
                            p_dst[k * c_blk + 0] = p_src[0][k];
                            p_dst[k * c_blk + 1] = p_src[1][k];
                            p_dst[k * c_blk + 2] = p_src[2][k];
                            p_dst[k * c_blk + 3] = p_src[3][k];
                        }
                    } else {
                        for (int64_t k = inner_dims_start; k < inner_dims_end; ++k) {
                            for (int64_t cc = 0; cc < c_eff; cc++) {
                                p_dst[k * c_blk + cc] = p_src[cc][k];
                            }
                            for (int64_t cc = c_eff; cc < c_blk; cc++) { // pad channels to zero
                                p_dst[k * c_blk + cc] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode N8cxToNdarrayFp16(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst)
{
    const int64_t simd_w     = 8;
    const int64_t c_blk      = 8;
    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c; c += c_blk) {
                const int64_t c_eff = min(channels - c, c_blk);
                float16x8_t vin[c_blk];
                float16x8_t vout[c_blk];

                const __fp16* p_src  = src + (n * pad_c + c) * inner_dims;
                __fp16* p_dst[c_blk] = {
                    dst + (n * channels + c + 0) * inner_dims,
                    dst + (n * channels + c + 1) * inner_dims,
                    dst + (n * channels + c + 2) * inner_dims,
                    dst + (n * channels + c + 3) * inner_dims,
                    dst + (n * channels + c + 4) * inner_dims,
                    dst + (n * channels + c + 5) * inner_dims,
                    dst + (n * channels + c + 6) * inner_dims,
                    dst + (n * channels + c + 7) * inner_dims};

                int64_t k = 0;
                if (c_eff == c_blk) {
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                        vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                        vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                        vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                        vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                        vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                        vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                        vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                        TRANSPOSE_8X8(vin, vout);
                        vst1q_f16(p_dst[0] + k, vout[0]);
                        vst1q_f16(p_dst[1] + k, vout[1]);
                        vst1q_f16(p_dst[2] + k, vout[2]);
                        vst1q_f16(p_dst[3] + k, vout[3]);
                        vst1q_f16(p_dst[4] + k, vout[4]);
                        vst1q_f16(p_dst[5] + k, vout[5]);
                        vst1q_f16(p_dst[6] + k, vout[6]);
                        vst1q_f16(p_dst[7] + k, vout[7]);
                    }
                } else {
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                        vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                        vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                        vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                        vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                        vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                        vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                        vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                        TRANSPOSE_8X8(vin, vout);
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            vst1q_f16(p_dst[cc] + k, vout[cc]);
                        }
                    }
                }
                for (; k < inner_dims; ++k) {
                    for (int64_t cc = 0; cc < c_eff; cc++) {
                        p_dst[cc][k] = p_src[k * c_blk + cc];
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(round_up(inner_dims, simd_w), num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c; c += c_blk) {
                    const int64_t c_eff = min(channels - c, c_blk);
                    float16x8_t vin[c_blk];
                    float16x8_t vout[c_blk];

                    const __fp16* p_src  = src + (n * pad_c + c) * inner_dims;
                    __fp16* p_dst[c_blk] = {
                        dst + (n * channels + c + 0) * inner_dims,
                        dst + (n * channels + c + 1) * inner_dims,
                        dst + (n * channels + c + 2) * inner_dims,
                        dst + (n * channels + c + 3) * inner_dims,
                        dst + (n * channels + c + 4) * inner_dims,
                        dst + (n * channels + c + 5) * inner_dims,
                        dst + (n * channels + c + 6) * inner_dims,
                        dst + (n * channels + c + 7) * inner_dims};

                    int64_t k = inner_dims_start;
                    if (c_eff == c_blk) {
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                            vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                            vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                            vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                            vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                            vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                            vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                            vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                            TRANSPOSE_8X8(vin, vout);
                            vst1q_f16(p_dst[0] + k, vout[0]);
                            vst1q_f16(p_dst[1] + k, vout[1]);
                            vst1q_f16(p_dst[2] + k, vout[2]);
                            vst1q_f16(p_dst[3] + k, vout[3]);
                            vst1q_f16(p_dst[4] + k, vout[4]);
                            vst1q_f16(p_dst[5] + k, vout[5]);
                            vst1q_f16(p_dst[6] + k, vout[6]);
                            vst1q_f16(p_dst[7] + k, vout[7]);
                        }
                    } else {
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                            vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                            vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                            vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                            vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                            vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                            vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                            vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                            TRANSPOSE_8X8(vin, vout);
                            for (int64_t cc = 0; cc < c_eff; cc++) {
                                vst1q_f16(p_dst[cc] + k, vout[cc]);
                            }
                        }
                    }
                    for (; k < inner_dims_end; ++k) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            p_dst[cc][k] = p_src[k * c_blk + cc];
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode NdarrayToN8cxFp16(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst)
{
    const int64_t simd_w     = 8;
    const int64_t c_blk      = 8;
    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    const float16x8_t v0 = vdupq_n_f16(0.0f);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c; c += c_blk) {
                const int64_t c_eff = min(channels - c, c_blk);
                float16x8_t vin[c_blk];
                float16x8_t vout[c_blk];

                const __fp16* p_src[c_blk] = {
                    src + (n * channels + c + 0) * inner_dims,
                    src + (n * channels + c + 1) * inner_dims,
                    src + (n * channels + c + 2) * inner_dims,
                    src + (n * channels + c + 3) * inner_dims,
                    src + (n * channels + c + 4) * inner_dims,
                    src + (n * channels + c + 5) * inner_dims,
                    src + (n * channels + c + 6) * inner_dims,
                    src + (n * channels + c + 7) * inner_dims};
                __fp16* p_dst = dst + (n * pad_c + c) * inner_dims;

                int64_t k = 0;
                if (c_eff == c_blk) {
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        vin[0] = vld1q_f16(p_src[0] + k);
                        vin[1] = vld1q_f16(p_src[1] + k);
                        vin[2] = vld1q_f16(p_src[2] + k);
                        vin[3] = vld1q_f16(p_src[3] + k);
                        vin[4] = vld1q_f16(p_src[4] + k);
                        vin[5] = vld1q_f16(p_src[5] + k);
                        vin[6] = vld1q_f16(p_src[6] + k);
                        vin[7] = vld1q_f16(p_src[7] + k);
                        TRANSPOSE_8X8(vin, vout);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                    }
                } else {
                    for (int64_t cc = c_eff; cc < c_blk; cc++) {
                        vin[cc] = v0;
                    }
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            vin[cc] = vld1q_f16(p_src[cc] + k);
                        }
                        TRANSPOSE_8X8(vin, vout);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                    }
                }
                for (; k < inner_dims; k++) {
                    for (int64_t cc = 0; cc < c_eff; cc++) {
                        p_dst[k * c_blk + cc] = p_src[cc][k];
                    }
                    for (int64_t cc = c_eff; cc < c_blk; cc++) { // pad channels to zero
                        p_dst[k * c_blk + cc] = 0;
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(round_up(inner_dims, simd_w), num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c; c += c_blk) {
                    const int64_t c_eff = min(channels - c, c_blk);
                    float16x8_t vin[c_blk];
                    float16x8_t vout[c_blk];

                    const __fp16* p_src[c_blk] = {
                        src + (n * channels + c + 0) * inner_dims,
                        src + (n * channels + c + 1) * inner_dims,
                        src + (n * channels + c + 2) * inner_dims,
                        src + (n * channels + c + 3) * inner_dims,
                        src + (n * channels + c + 4) * inner_dims,
                        src + (n * channels + c + 5) * inner_dims,
                        src + (n * channels + c + 6) * inner_dims,
                        src + (n * channels + c + 7) * inner_dims};
                    __fp16* p_dst = dst + (n * pad_c + c) * inner_dims;

                    int64_t k = inner_dims_start;
                    if (c_eff == c_blk) {
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            vin[0] = vld1q_f16(p_src[0] + k);
                            vin[1] = vld1q_f16(p_src[1] + k);
                            vin[2] = vld1q_f16(p_src[2] + k);
                            vin[3] = vld1q_f16(p_src[3] + k);
                            vin[4] = vld1q_f16(p_src[4] + k);
                            vin[5] = vld1q_f16(p_src[5] + k);
                            vin[6] = vld1q_f16(p_src[6] + k);
                            vin[7] = vld1q_f16(p_src[7] + k);
                            TRANSPOSE_8X8(vin, vout);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                        }
                    } else {
                        for (int64_t cc = c_eff; cc < c_blk; cc++) {
                            vin[cc] = v0;
                        }
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            for (int64_t cc = 0; cc < c_eff; cc++) {
                                vin[cc] = vld1q_f16(p_src[cc] + k);
                            }
                            TRANSPOSE_8X8(vin, vout);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                        }
                    }
                    for (; k < inner_dims_end; k++) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            p_dst[k * c_blk + cc] = p_src[cc][k];
                        }
                        for (int64_t cc = c_eff; cc < c_blk; cc++) { // pad channels to zero
                            p_dst[k * c_blk + cc] = 0;
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline float16x8_t load_and_cvt_fp32x8(const float* data)
{
    const float32x4_t v_fp32_0 = vld1q_f32(data + 0);
    const float32x4_t v_fp32_1 = vld1q_f32(data + 4);
    return vcombine_f16(vcvt_f16_f32(v_fp32_0), vcvt_f16_f32(v_fp32_1));
}

inline void cvt_and_store_fp16x8(float* data, const float16x8_t v_fp16)
{
    const float32x4_t v_fp32_0 = vcvt_f32_f16(vget_low_f16(v_fp16));
    const float32x4_t v_fp32_1 = vcvt_f32_f16(vget_high_f16(v_fp16));
    vst1q_f32(data + 0, v_fp32_0);
    vst1q_f32(data + 4, v_fp32_1);
}

ppl::common::RetCode NdarrayFp32ToN8cxFp16(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst)
{
    const int64_t simd_w     = 8;
    const int64_t c_blk      = 8;
    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    const float16x8_t v0 = vdupq_n_f16(0.0f);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c; c += c_blk) {
                const int64_t c_eff = min(channels - c, c_blk);
                float16x8_t vin[c_blk];
                float16x8_t vout[c_blk];

                const float* p_src[c_blk] = {
                    src + (n * channels + c + 0) * inner_dims,
                    src + (n * channels + c + 1) * inner_dims,
                    src + (n * channels + c + 2) * inner_dims,
                    src + (n * channels + c + 3) * inner_dims,
                    src + (n * channels + c + 4) * inner_dims,
                    src + (n * channels + c + 5) * inner_dims,
                    src + (n * channels + c + 6) * inner_dims,
                    src + (n * channels + c + 7) * inner_dims};
                __fp16* p_dst = dst + (n * pad_c + c) * inner_dims;

                int64_t k = 0;
                if (c_eff == c_blk) {
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        vin[0] = load_and_cvt_fp32x8(p_src[0] + k);
                        vin[1] = load_and_cvt_fp32x8(p_src[1] + k);
                        vin[2] = load_and_cvt_fp32x8(p_src[2] + k);
                        vin[3] = load_and_cvt_fp32x8(p_src[3] + k);
                        vin[4] = load_and_cvt_fp32x8(p_src[4] + k);
                        vin[5] = load_and_cvt_fp32x8(p_src[5] + k);
                        vin[6] = load_and_cvt_fp32x8(p_src[6] + k);
                        vin[7] = load_and_cvt_fp32x8(p_src[7] + k);
                        TRANSPOSE_8X8(vin, vout);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                    }
                } else {
                    for (int64_t cc = c_eff; cc < c_blk; cc++) {
                        vin[cc] = v0;
                    }
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            vin[cc] = load_and_cvt_fp32x8(p_src[cc] + k);
                        }
                        TRANSPOSE_8X8(vin, vout);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                        vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                    }
                }
                for (; k < inner_dims; k++) {
                    for (int64_t cc = 0; cc < c_eff; cc++) {
                        p_dst[k * c_blk + cc] = p_src[cc][k];
                    }
                    for (int64_t cc = c_eff; cc < c_blk; cc++) { // pad channels to zero
                        p_dst[k * c_blk + cc] = 0;
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(round_up(inner_dims, simd_w), num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c; c += c_blk) {
                    const int64_t c_eff = min(channels - c, c_blk);
                    float16x8_t vin[c_blk];
                    float16x8_t vout[c_blk];

                    const float* p_src[c_blk] = {
                        src + (n * channels + c + 0) * inner_dims,
                        src + (n * channels + c + 1) * inner_dims,
                        src + (n * channels + c + 2) * inner_dims,
                        src + (n * channels + c + 3) * inner_dims,
                        src + (n * channels + c + 4) * inner_dims,
                        src + (n * channels + c + 5) * inner_dims,
                        src + (n * channels + c + 6) * inner_dims,
                        src + (n * channels + c + 7) * inner_dims};
                    __fp16* p_dst = dst + (n * pad_c + c) * inner_dims;

                    int64_t k = inner_dims_start;
                    if (c_eff == c_blk) {
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            vin[0] = load_and_cvt_fp32x8(p_src[0] + k);
                            vin[1] = load_and_cvt_fp32x8(p_src[1] + k);
                            vin[2] = load_and_cvt_fp32x8(p_src[2] + k);
                            vin[3] = load_and_cvt_fp32x8(p_src[3] + k);
                            vin[4] = load_and_cvt_fp32x8(p_src[4] + k);
                            vin[5] = load_and_cvt_fp32x8(p_src[5] + k);
                            vin[6] = load_and_cvt_fp32x8(p_src[6] + k);
                            vin[7] = load_and_cvt_fp32x8(p_src[7] + k);
                            TRANSPOSE_8X8(vin, vout);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                        }
                    } else {
                        for (int64_t cc = c_eff; cc < c_blk; cc++) {
                            vin[cc] = v0;
                        }
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            for (int64_t cc = 0; cc < c_eff; cc++) {
                                vin[cc] = load_and_cvt_fp32x8(p_src[cc] + k);
                            }
                            TRANSPOSE_8X8(vin, vout);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 0, vout[0]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 1, vout[1]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 2, vout[2]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 3, vout[3]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 4, vout[4]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 5, vout[5]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 6, vout[6]);
                            vst1q_f16(p_dst + k * c_blk + simd_w * 7, vout[7]);
                        }
                    }
                    for (; k < inner_dims_end; k++) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            p_dst[k * c_blk + cc] = p_src[cc][k];
                        }
                        for (int64_t cc = c_eff; cc < c_blk; cc++) { // pad channels to zero
                            p_dst[k * c_blk + cc] = 0;
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode N8cxFp16ToNdarrayFp32(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst)
{
    const int64_t simd_w     = 8;
    const int64_t c_blk      = 8;
    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c; c += c_blk) {
                const int64_t c_eff = min(channels - c, c_blk);
                float16x8_t vin[c_blk];
                float16x8_t vout[c_blk];

                const __fp16* p_src = src + (n * pad_c + c) * inner_dims;
                float* p_dst[c_blk] = {
                    dst + (n * channels + c + 0) * inner_dims,
                    dst + (n * channels + c + 1) * inner_dims,
                    dst + (n * channels + c + 2) * inner_dims,
                    dst + (n * channels + c + 3) * inner_dims,
                    dst + (n * channels + c + 4) * inner_dims,
                    dst + (n * channels + c + 5) * inner_dims,
                    dst + (n * channels + c + 6) * inner_dims,
                    dst + (n * channels + c + 7) * inner_dims};

                int64_t k = 0;
                if (c_eff == c_blk) {
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                        vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                        vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                        vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                        vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                        vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                        vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                        vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                        TRANSPOSE_8X8(vin, vout);
                        cvt_and_store_fp16x8(p_dst[0] + k, vout[0]);
                        cvt_and_store_fp16x8(p_dst[1] + k, vout[1]);
                        cvt_and_store_fp16x8(p_dst[2] + k, vout[2]);
                        cvt_and_store_fp16x8(p_dst[3] + k, vout[3]);
                        cvt_and_store_fp16x8(p_dst[4] + k, vout[4]);
                        cvt_and_store_fp16x8(p_dst[5] + k, vout[5]);
                        cvt_and_store_fp16x8(p_dst[6] + k, vout[6]);
                        cvt_and_store_fp16x8(p_dst[7] + k, vout[7]);
                    }
                } else {
                    for (; k + simd_w <= inner_dims; k += simd_w) {
                        vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                        vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                        vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                        vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                        vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                        vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                        vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                        vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                        TRANSPOSE_8X8(vin, vout);
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            cvt_and_store_fp16x8(p_dst[cc] + k, vout[cc]);
                        }
                    }
                }
                for (; k < inner_dims; k++) {
                    for (int64_t cc = 0; cc < c_eff; cc++) {
                        p_dst[cc][k] = p_src[k * c_blk + cc];
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(round_up(inner_dims, simd_w), num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c; c += c_blk) {
                    const int64_t c_eff = min(channels - c, c_blk);
                    float16x8_t vin[c_blk];
                    float16x8_t vout[c_blk];

                    const __fp16* p_src = src + (n * pad_c + c) * inner_dims;
                    float* p_dst[c_blk] = {
                        dst + (n * channels + c + 0) * inner_dims,
                        dst + (n * channels + c + 1) * inner_dims,
                        dst + (n * channels + c + 2) * inner_dims,
                        dst + (n * channels + c + 3) * inner_dims,
                        dst + (n * channels + c + 4) * inner_dims,
                        dst + (n * channels + c + 5) * inner_dims,
                        dst + (n * channels + c + 6) * inner_dims,
                        dst + (n * channels + c + 7) * inner_dims};

                    int64_t k = inner_dims_start;
                    if (c_eff == c_blk) {
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                            vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                            vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                            vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                            vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                            vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                            vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                            vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                            TRANSPOSE_8X8(vin, vout);
                            cvt_and_store_fp16x8(p_dst[0] + k, vout[0]);
                            cvt_and_store_fp16x8(p_dst[1] + k, vout[1]);
                            cvt_and_store_fp16x8(p_dst[2] + k, vout[2]);
                            cvt_and_store_fp16x8(p_dst[3] + k, vout[3]);
                            cvt_and_store_fp16x8(p_dst[4] + k, vout[4]);
                            cvt_and_store_fp16x8(p_dst[5] + k, vout[5]);
                            cvt_and_store_fp16x8(p_dst[6] + k, vout[6]);
                            cvt_and_store_fp16x8(p_dst[7] + k, vout[7]);
                        }
                    } else {
                        for (; k + simd_w <= inner_dims_end; k += simd_w) {
                            vin[0] = vld1q_f16(p_src + k * c_blk + simd_w * 0);
                            vin[1] = vld1q_f16(p_src + k * c_blk + simd_w * 1);
                            vin[2] = vld1q_f16(p_src + k * c_blk + simd_w * 2);
                            vin[3] = vld1q_f16(p_src + k * c_blk + simd_w * 3);
                            vin[4] = vld1q_f16(p_src + k * c_blk + simd_w * 4);
                            vin[5] = vld1q_f16(p_src + k * c_blk + simd_w * 5);
                            vin[6] = vld1q_f16(p_src + k * c_blk + simd_w * 6);
                            vin[7] = vld1q_f16(p_src + k * c_blk + simd_w * 7);
                            TRANSPOSE_8X8(vin, vout);
                            for (int64_t cc = 0; cc < c_eff; cc++) {
                                cvt_and_store_fp16x8(p_dst[cc] + k, vout[cc]);
                            }
                        }
                    }
                    for (; k < inner_dims_end; k++) {
                        for (int64_t cc = 0; cc < c_eff; cc++) {
                            p_dst[cc][k] = p_src[k * c_blk + cc];
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode Fp32ToFp16(const float* src, const int64_t len, __fp16* dst)
{
    const int64_t simd_w      = 8;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(len, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        const float16x8_t v_data_0 = load_and_cvt_fp32x8(src + i + simd_w * 0);
        const float16x8_t v_data_1 = load_and_cvt_fp32x8(src + i + simd_w * 1);
        const float16x8_t v_data_2 = load_and_cvt_fp32x8(src + i + simd_w * 2);
        const float16x8_t v_data_3 = load_and_cvt_fp32x8(src + i + simd_w * 3);
        vst1q_f16(dst + i + simd_w * 0, v_data_0);
        vst1q_f16(dst + i + simd_w * 1, v_data_1);
        vst1q_f16(dst + i + simd_w * 2, v_data_2);
        vst1q_f16(dst + i + simd_w * 3, v_data_3);
    }
    for (int64_t i = unroll_body; i < len; i++) {
        dst[i] = src[i];
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode Fp16ToFp32(const __fp16* src, const int64_t len, float* dst)
{
    const int64_t simd_w      = 8;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(len, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        const float16x8_t v_data_0 = vld1q_f16(src + i + simd_w * 0);
        const float16x8_t v_data_1 = vld1q_f16(src + i + simd_w * 1);
        const float16x8_t v_data_2 = vld1q_f16(src + i + simd_w * 2);
        const float16x8_t v_data_3 = vld1q_f16(src + i + simd_w * 3);
        cvt_and_store_fp16x8(dst + i + simd_w * 0, v_data_0);
        cvt_and_store_fp16x8(dst + i + simd_w * 1, v_data_1);
        cvt_and_store_fp16x8(dst + i + simd_w * 2, v_data_2);
        cvt_and_store_fp16x8(dst + i + simd_w * 3, v_data_3);
    }
    for (int64_t i = unroll_body; i < len; i++) {
        dst[i] = src[i];
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode N4cxFp32ToN8cxFp16(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst)
{
    const int64_t simd_w_fp32 = 4;
    const int64_t simd_w_fp16 = 8;
    const int64_t c_blk_in    = 4;
    const int64_t c_blk_out   = 8;
    const int64_t pad_c_in    = round_up(channels, c_blk_in);
    const int64_t pad_c_out   = round_up(channels, c_blk_out);
    const int64_t inner_dims  = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk_out), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c_out; c += c_blk_out) {
                const int64_t c_eff_in = min(pad_c_in - c, c_blk_out);
                if (c_eff_in == c_blk_out) {
                    const float* p_src_0 = src + (n * pad_c_in + c) * inner_dims;
                    const float* p_src_1 = p_src_0 + c_blk_in * inner_dims;
                    __fp16* p_dst        = dst + (n * pad_c_out + c) * inner_dims;
                    for (int64_t hw = 0; hw < inner_dims; hw++) {
                        const float32x4_t v_src_0 = vld1q_f32(p_src_0 + hw * c_blk_in);
                        const float32x4_t v_src_1 = vld1q_f32(p_src_1 + hw * c_blk_in);
                        const float16x8_t v_dst   = vcombine_f16(vcvt_f16_f32(v_src_0), vcvt_f16_f32(v_src_1));
                        vst1q_f16(p_dst + hw * c_blk_out, v_dst);
                    }
                } else if (c_eff_in == c_blk_in) {
                    const float* p_src       = src + (n * pad_c_in + c) * inner_dims;
                    __fp16* p_dst            = dst + (n * pad_c_out + c) * inner_dims;
                    const float16x4_t v_zero = vdup_n_f16(0);
                    for (int64_t hw = 0; hw < inner_dims; hw++) {
                        const float32x4_t v_src = vld1q_f32(p_src + hw * c_blk_in);
                        const float16x8_t v_dst = vcombine_f16(vcvt_f16_f32(v_src), v_zero);
                        vst1q_f16(p_dst + hw * c_blk_out, v_dst);
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(inner_dims, num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c_out; c += c_blk_out) {
                    const int64_t c_eff_in = min(pad_c_in - c, c_blk_out);
                    if (c_eff_in == c_blk_out) {
                        const float* p_src_0 = src + (n * pad_c_in + c) * inner_dims;
                        const float* p_src_1 = p_src_0 + c_blk_in * inner_dims;
                        __fp16* p_dst        = dst + (n * pad_c_out + c) * inner_dims;
                        for (int64_t hw = inner_dims_start; hw < inner_dims_end; hw++) {
                            const float32x4_t v_src_0 = vld1q_f32(p_src_0 + hw * c_blk_in);
                            const float32x4_t v_src_1 = vld1q_f32(p_src_1 + hw * c_blk_in);
                            const float16x8_t v_dst   = vcombine_f16(vcvt_f16_f32(v_src_0), vcvt_f16_f32(v_src_1));
                            vst1q_f16(p_dst + hw * c_blk_out, v_dst);
                        }
                    } else if (c_eff_in == c_blk_in) {
                        const float* p_src       = src + (n * pad_c_in + c) * inner_dims;
                        __fp16* p_dst            = dst + (n * pad_c_out + c) * inner_dims;
                        const float16x4_t v_zero = vdup_n_f16(0);
                        for (int64_t hw = inner_dims_start; hw < inner_dims_end; hw++) {
                            const float32x4_t v_src = vld1q_f32(p_src + hw * c_blk_in);
                            const float16x8_t v_dst = vcombine_f16(vcvt_f16_f32(v_src), v_zero);
                            vst1q_f16(p_dst + hw * c_blk_out, v_dst);
                        }
                    }
                }
            }
        }
    }

    (void)simd_w_fp16;
    (void)simd_w_fp32;

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode N8cxFp16ToN4cxFp32(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst)
{
    const int64_t simd_w_fp16 = 8;
    const int64_t simd_w_fp32 = 4;
    const int64_t c_blk_in    = 8;
    const int64_t c_blk_out   = 4;
    const int64_t pad_c_in    = round_up(channels, c_blk_in);
    const int64_t pad_c_out   = round_up(channels, c_blk_out);
    const int64_t inner_dims  = height * width;

    std::vector<int64_t> iter_of_loop{batch * div_up(channels, c_blk_out), inner_dims};
    const float omp_div_task_time_ratio  = 20.0f; // assume that omp create thread is 20x slower than element copy
    single_parallel_loop_config_t config = select_single_parallel_loop(iter_of_loop, omp_div_task_time_ratio);

    if (config.depth_of_loop == 0) { // divide by outer dims is better
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < pad_c_in; c += c_blk_in) {
                const int64_t c_eff_out = min(pad_c_out - c, c_blk_in);
                if (c_eff_out == c_blk_in) {
                    const __fp16* p_src = src + (n * pad_c_in + c) * inner_dims;
                    float* p_dst_0      = dst + (n * pad_c_out + c) * inner_dims;
                    float* p_dst_1      = p_dst_0 + c_blk_out * inner_dims;
                    for (int64_t hw = 0; hw < inner_dims; hw++) {
                        const float16x8_t v_src   = vld1q_f16(p_src + hw * c_blk_in);
                        const float32x4_t v_dst_0 = vcvt_f32_f16(vget_low_f16(v_src));
                        const float32x4_t v_dst_1 = vcvt_f32_f16(vget_high_f16(v_src));
                        vst1q_f32(p_dst_0 + hw * c_blk_out, v_dst_0);
                        vst1q_f32(p_dst_1 + hw * c_blk_out, v_dst_1);
                    }
                } else if (c_eff_out == c_blk_out) {
                    const __fp16* p_src = src + (n * pad_c_in + c) * inner_dims;
                    float* p_dst        = dst + (n * pad_c_out + c) * inner_dims;
                    for (int64_t hw = 0; hw < inner_dims; hw++) {
                        const float16x8_t v_src = vld1q_f16(p_src + hw * c_blk_in);
                        const float32x4_t v_dst = vcvt_f32_f16(vget_low_f16(v_src));
                        vst1q_f32(p_dst + hw * c_blk_out, v_dst);
                    }
                }
            }
        }
    } else { // divide by inner dims is better
        const int64_t num_threads           = config.num_threads;
        const int64_t inner_dims_per_thread = div_up(inner_dims, num_threads);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t inner_dims_start = inner_dims_per_thread * thread_id;
            const int64_t inner_dims_end   = min(inner_dims_start + inner_dims_per_thread, inner_dims);
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t c = 0; c < pad_c_in; c += c_blk_in) {
                    const int64_t c_eff_out = min(pad_c_out - c, c_blk_in);
                    if (c_eff_out == c_blk_in) {
                        const __fp16* p_src = src + (n * pad_c_in + c) * inner_dims;
                        float* p_dst_0      = dst + (n * pad_c_out + c) * inner_dims;
                        float* p_dst_1      = p_dst_0 + c_blk_out * inner_dims;
                        for (int64_t hw = inner_dims_start; hw < inner_dims_end; hw++) {
                            const float16x8_t v_src   = vld1q_f16(p_src + hw * c_blk_in);
                            const float32x4_t v_dst_0 = vcvt_f32_f16(vget_low_f16(v_src));
                            const float32x4_t v_dst_1 = vcvt_f32_f16(vget_high_f16(v_src));
                            vst1q_f32(p_dst_0 + hw * c_blk_out, v_dst_0);
                            vst1q_f32(p_dst_1 + hw * c_blk_out, v_dst_1);
                        }
                    } else if (c_eff_out == c_blk_out) {
                        const __fp16* p_src = src + (n * pad_c_in + c) * inner_dims;
                        float* p_dst        = dst + (n * pad_c_out + c) * inner_dims;
                        for (int64_t hw = inner_dims_start; hw < inner_dims_end; hw++) {
                            const float16x8_t v_src = vld1q_f16(p_src + hw * c_blk_in);
                            const float32x4_t v_dst = vcvt_f32_f16(vget_low_f16(v_src));
                            vst1q_f32(p_dst + hw * c_blk_out, v_dst);
                        }
                    }
                }
            }
        }
    }

    (void)simd_w_fp16;
    (void)simd_w_fp32;

    return ppl::common::RC_SUCCESS;
}
#endif

}}}; // namespace ppl::kernel::arm_server
