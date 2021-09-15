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

#include <new>
#include <immintrin.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_depthwise_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_depthwise_kernel_fp32_sse.h"
#include "ppl/kernel/x86/fp32/transpose/sse/transpose_fp32_sse.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

void conv2d_depthwise_fp32_sse_executor::init_preproc_param()
{
    schedule_param_.padded_ch = round_up(conv_param_->group, CH_DT_BLK());
    schedule_param_.ow_kr_blk = MAX_OW_RF();
}

void conv2d_depthwise_fp32_sse_executor::cal_kernel_tunning_param()
{
}

uint64_t conv2d_depthwise_fp32_sse_executor::cal_temp_buffer_size()
{
    const int64_t src_h         = src_shape_->GetDim(2);
    const int64_t src_w         = src_shape_->GetDim(3);
    const int64_t padded_src_w  = src_w + 2 * conv_param_->pad_w;
    const int64_t dst_w         = dst_shape_->GetDim(3);
    const int64_t src_trans_len = src_h * padded_src_w * CH_DT_BLK();
    const int64_t dst_buf_len   = dst_w * CH_DT_BLK();
    return ((uint64_t)src_trans_len + dst_buf_len) * PPL_OMP_MAX_THREADS() * sizeof(float);
}

ppl::common::RetCode conv2d_depthwise_fp32_sse_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

template<int32_t relu, int32_t sum>
static void conv2d_depthwise_fp32_sse_dst_trans(
    const float *dst_trans,
    const float *sum_src,
    int64_t width,
    int64_t channels,
    int64_t dst_c_stride,
    float *dst)
{
    const float *l_dst_trans = dst_trans;
    const float *l_sum = sum_src;
    float *l_dst = dst;
    const int64_t unroll_w = 4;
    int64_t w = width;
    if (channels == CH_DT_BLK()) {
        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
        __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13;
        if (relu >= 1) xmm12 = _mm_set1_ps(0.0f);
        if (relu == 6) xmm13 = _mm_set1_ps(6.0f);
        while (w >= unroll_w * 2) {
            w -= unroll_w * 2;
            xmm0 = _mm_loadu_ps(l_dst_trans + 0 * CH_DT_BLK());
            xmm1 = _mm_loadu_ps(l_dst_trans + 1 * CH_DT_BLK());
            xmm2 = _mm_loadu_ps(l_dst_trans + 2 * CH_DT_BLK());
            xmm3 = _mm_loadu_ps(l_dst_trans + 3 * CH_DT_BLK());
            xmm4 = _mm_loadu_ps(l_dst_trans + 4 * CH_DT_BLK());
            xmm5 = _mm_loadu_ps(l_dst_trans + 5 * CH_DT_BLK());
            xmm6 = _mm_loadu_ps(l_dst_trans + 6 * CH_DT_BLK());
            xmm7 = _mm_loadu_ps(l_dst_trans + 7 * CH_DT_BLK());
            l_dst_trans += 2 * unroll_w * CH_DT_BLK();

            TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm8, xmm9, xmm10, xmm11);
            TRANSPOSE_4X4_FP32_SSE_MACRO(xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11);

            if (sum) {
                xmm0 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 0 * dst_c_stride), xmm0);
                xmm1 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 1 * dst_c_stride), xmm1);
                xmm2 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 2 * dst_c_stride), xmm2);
                xmm3 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 3 * dst_c_stride), xmm3);
                xmm4 = _mm_add_ps(_mm_loadu_ps(l_sum + 1 * unroll_w + 0 * dst_c_stride), xmm4);
                xmm5 = _mm_add_ps(_mm_loadu_ps(l_sum + 1 * unroll_w + 1 * dst_c_stride), xmm5);
                xmm6 = _mm_add_ps(_mm_loadu_ps(l_sum + 1 * unroll_w + 2 * dst_c_stride), xmm6);
                xmm7 = _mm_add_ps(_mm_loadu_ps(l_sum + 1 * unroll_w + 3 * dst_c_stride), xmm7);
                l_sum += 2 * unroll_w;
            }

            if (relu >= 1) {
                xmm0 = _mm_max_ps(xmm0, xmm12);
                xmm1 = _mm_max_ps(xmm1, xmm12);
                xmm2 = _mm_max_ps(xmm2, xmm12);
                xmm3 = _mm_max_ps(xmm3, xmm12);
                xmm4 = _mm_max_ps(xmm4, xmm12);
                xmm5 = _mm_max_ps(xmm5, xmm12);
                xmm6 = _mm_max_ps(xmm6, xmm12);
                xmm7 = _mm_max_ps(xmm7, xmm12);
            }

            if (relu == 6) {
                xmm0 = _mm_min_ps(xmm0, xmm13);
                xmm1 = _mm_min_ps(xmm1, xmm13);
                xmm2 = _mm_min_ps(xmm2, xmm13);
                xmm3 = _mm_min_ps(xmm3, xmm13);
                xmm4 = _mm_min_ps(xmm4, xmm13);
                xmm5 = _mm_min_ps(xmm5, xmm13);
                xmm6 = _mm_min_ps(xmm6, xmm13);
                xmm7 = _mm_min_ps(xmm7, xmm13);
            }

            _mm_storeu_ps(l_dst + 0 * unroll_w + 0 * dst_c_stride, xmm0);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 1 * dst_c_stride, xmm1);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 2 * dst_c_stride, xmm2);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 3 * dst_c_stride, xmm3);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 0 * dst_c_stride, xmm4);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 1 * dst_c_stride, xmm5);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 2 * dst_c_stride, xmm6);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 3 * dst_c_stride, xmm7);
            l_dst += 2 * unroll_w;
        }
        if (w & 4) {
            xmm0 = _mm_loadu_ps(l_dst_trans + 0 * CH_DT_BLK());
            xmm1 = _mm_loadu_ps(l_dst_trans + 1 * CH_DT_BLK());
            xmm2 = _mm_loadu_ps(l_dst_trans + 2 * CH_DT_BLK());
            xmm3 = _mm_loadu_ps(l_dst_trans + 3 * CH_DT_BLK());
            l_dst_trans += 1 * unroll_w * CH_DT_BLK();

            TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm8, xmm9, xmm10, xmm11);

            if (sum) {
                xmm0 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 0 * dst_c_stride), xmm0);
                xmm1 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 1 * dst_c_stride), xmm1);
                xmm2 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 2 * dst_c_stride), xmm2);
                xmm3 = _mm_add_ps(_mm_loadu_ps(l_sum + 0 * unroll_w + 3 * dst_c_stride), xmm3);
                l_sum += 1 * unroll_w;
            }

            if (relu >= 1) {
                xmm0 = _mm_max_ps(xmm0, xmm12);
                xmm1 = _mm_max_ps(xmm1, xmm12);
                xmm2 = _mm_max_ps(xmm2, xmm12);
                xmm3 = _mm_max_ps(xmm3, xmm12);
            }

            if (relu == 6) {
                xmm0 = _mm_min_ps(xmm0, xmm13);
                xmm1 = _mm_min_ps(xmm1, xmm13);
                xmm2 = _mm_min_ps(xmm2, xmm13);
                xmm3 = _mm_min_ps(xmm3, xmm13);
            }

            _mm_storeu_ps(l_dst + 0 * unroll_w + 0 * dst_c_stride, xmm0);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 1 * dst_c_stride, xmm1);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 2 * dst_c_stride, xmm2);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 3 * dst_c_stride, xmm3);
            l_dst += 1 * unroll_w;
        }
        if (w & 2) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c1w1 = l_dst_trans[1 * CH_DT_BLK() + 1];
            float c2w0 = l_dst_trans[0 * CH_DT_BLK() + 2];
            float c2w1 = l_dst_trans[1 * CH_DT_BLK() + 2];
            float c3w0 = l_dst_trans[0 * CH_DT_BLK() + 3];
            float c3w1 = l_dst_trans[1 * CH_DT_BLK() + 3];
            l_dst_trans += 2 * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c1w1 += l_sum[1 * dst_c_stride + 1];
                c2w0 += l_sum[2 * dst_c_stride + 0];
                c2w1 += l_sum[2 * dst_c_stride + 1];
                c3w0 += l_sum[3 * dst_c_stride + 0];
                c3w1 += l_sum[3 * dst_c_stride + 1];
                l_sum += 2;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c1w1 = max(c1w1, 0.0f);
                c2w0 = max(c2w0, 0.0f);
                c2w1 = max(c2w1, 0.0f);
                c3w0 = max(c3w0, 0.0f);
                c3w1 = max(c3w1, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c1w1 = min(c1w1, 6.0f);
                c2w0 = min(c2w0, 6.0f);
                c2w1 = min(c2w1, 6.0f);
                c3w0 = min(c3w0, 6.0f);
                c3w1 = min(c3w1, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[1 * dst_c_stride + 1] = c1w1;
            l_dst[2 * dst_c_stride + 0] = c2w0;
            l_dst[2 * dst_c_stride + 1] = c2w1;
            l_dst[3 * dst_c_stride + 0] = c3w0;
            l_dst[3 * dst_c_stride + 1] = c3w1;
            l_dst += 2;
        }
        if (w & 1) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c2w0 = l_dst_trans[0 * CH_DT_BLK() + 2];
            float c3w0 = l_dst_trans[0 * CH_DT_BLK() + 3];

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c2w0 += l_sum[2 * dst_c_stride + 0];
                c3w0 += l_sum[3 * dst_c_stride + 0];
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c2w0 = max(c2w0, 0.0f);
                c3w0 = max(c3w0, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c2w0 = min(c2w0, 6.0f);
                c3w0 = min(c3w0, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[2 * dst_c_stride + 0] = c2w0;
            l_dst[3 * dst_c_stride + 0] = c3w0;
        }
    } else if (channels == 3) {
        while (w >= unroll_w) {
            w -= unroll_w;
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            float c0w2 = l_dst_trans[2 * CH_DT_BLK() + 0];
            float c0w3 = l_dst_trans[3 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c1w1 = l_dst_trans[1 * CH_DT_BLK() + 1];
            float c1w2 = l_dst_trans[2 * CH_DT_BLK() + 1];
            float c1w3 = l_dst_trans[3 * CH_DT_BLK() + 1];
            float c2w0 = l_dst_trans[0 * CH_DT_BLK() + 2];
            float c2w1 = l_dst_trans[1 * CH_DT_BLK() + 2];
            float c2w2 = l_dst_trans[2 * CH_DT_BLK() + 2];
            float c2w3 = l_dst_trans[3 * CH_DT_BLK() + 2];
            l_dst_trans += unroll_w * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                c0w2 += l_sum[0 * dst_c_stride + 2];
                c0w3 += l_sum[0 * dst_c_stride + 3];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c1w1 += l_sum[1 * dst_c_stride + 1];
                c1w2 += l_sum[1 * dst_c_stride + 2];
                c1w2 += l_sum[1 * dst_c_stride + 3];
                c2w0 += l_sum[2 * dst_c_stride + 0];
                c2w1 += l_sum[2 * dst_c_stride + 1];
                c2w2 += l_sum[2 * dst_c_stride + 2];
                c2w3 += l_sum[2 * dst_c_stride + 3];
                l_sum += unroll_w;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
                c0w2 = max(c0w2, 0.0f);
                c0w3 = max(c0w3, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c1w1 = max(c1w1, 0.0f);
                c1w2 = max(c1w2, 0.0f);
                c1w3 = max(c1w3, 0.0f);
                c2w0 = max(c2w0, 0.0f);
                c2w1 = max(c2w1, 0.0f);
                c2w2 = max(c2w2, 0.0f);
                c2w3 = max(c2w3, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
                c0w2 = min(c0w2, 6.0f);
                c0w3 = min(c0w3, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c1w1 = min(c1w1, 6.0f);
                c1w2 = min(c1w2, 6.0f);
                c1w3 = min(c1w3, 6.0f);
                c2w0 = min(c2w0, 6.0f);
                c2w1 = min(c2w1, 6.0f);
                c2w2 = min(c2w2, 6.0f);
                c2w3 = min(c2w3, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst[0 * dst_c_stride + 2] = c0w2;
            l_dst[0 * dst_c_stride + 3] = c0w3;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[1 * dst_c_stride + 1] = c1w1;
            l_dst[1 * dst_c_stride + 2] = c1w2;
            l_dst[1 * dst_c_stride + 3] = c1w3;
            l_dst[2 * dst_c_stride + 0] = c2w0;
            l_dst[2 * dst_c_stride + 1] = c2w1;
            l_dst[2 * dst_c_stride + 2] = c2w2;
            l_dst[2 * dst_c_stride + 3] = c2w3;
            l_dst += unroll_w;
        }
        if (w & 2) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c1w1 = l_dst_trans[1 * CH_DT_BLK() + 1];
            float c2w0 = l_dst_trans[0 * CH_DT_BLK() + 2];
            float c2w1 = l_dst_trans[1 * CH_DT_BLK() + 2];
            l_dst_trans += 2 * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c1w1 += l_sum[1 * dst_c_stride + 1];
                c2w0 += l_sum[2 * dst_c_stride + 0];
                c2w1 += l_sum[2 * dst_c_stride + 1];
                l_sum += 2;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c1w1 = max(c1w1, 0.0f);
                c2w0 = max(c2w0, 0.0f);
                c2w1 = max(c2w1, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c1w1 = min(c1w1, 6.0f);
                c2w0 = min(c2w0, 6.0f);
                c2w1 = min(c2w1, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[1 * dst_c_stride + 1] = c1w1;
            l_dst[2 * dst_c_stride + 0] = c2w0;
            l_dst[2 * dst_c_stride + 1] = c2w1;
            l_dst += 2;
        }
        if (w & 1) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c2w0 = l_dst_trans[0 * CH_DT_BLK() + 2];

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c2w0 += l_sum[2 * dst_c_stride + 0];
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c2w0 = max(c2w0, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c2w0 = min(c2w0, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[2 * dst_c_stride + 0] = c2w0;
        }
    } else if (channels == 2) {
        while (w >= unroll_w) {
            w -= unroll_w;
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            float c0w2 = l_dst_trans[2 * CH_DT_BLK() + 0];
            float c0w3 = l_dst_trans[3 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c1w1 = l_dst_trans[1 * CH_DT_BLK() + 1];
            float c1w2 = l_dst_trans[2 * CH_DT_BLK() + 1];
            float c1w3 = l_dst_trans[3 * CH_DT_BLK() + 1];
            l_dst_trans += unroll_w * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                c0w2 += l_sum[0 * dst_c_stride + 2];
                c0w3 += l_sum[0 * dst_c_stride + 3];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c1w1 += l_sum[1 * dst_c_stride + 1];
                c1w2 += l_sum[1 * dst_c_stride + 2];
                c1w2 += l_sum[1 * dst_c_stride + 3];
                l_sum += unroll_w;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
                c0w2 = max(c0w2, 0.0f);
                c0w3 = max(c0w3, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c1w1 = max(c1w1, 0.0f);
                c1w2 = max(c1w2, 0.0f);
                c1w3 = max(c1w3, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
                c0w2 = min(c0w2, 6.0f);
                c0w3 = min(c0w3, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c1w1 = min(c1w1, 6.0f);
                c1w2 = min(c1w2, 6.0f);
                c1w3 = min(c1w3, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst[0 * dst_c_stride + 2] = c0w2;
            l_dst[0 * dst_c_stride + 3] = c0w3;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[1 * dst_c_stride + 1] = c1w1;
            l_dst[1 * dst_c_stride + 2] = c1w2;
            l_dst[1 * dst_c_stride + 3] = c1w3;
            l_dst += unroll_w;
        }
        if (w & 2) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];
            float c1w1 = l_dst_trans[1 * CH_DT_BLK() + 1];
            l_dst_trans += 2 * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                c1w0 += l_sum[1 * dst_c_stride + 0];
                c1w1 += l_sum[1 * dst_c_stride + 1];
                l_sum += 2;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
                c1w0 = max(c1w0, 0.0f);
                c1w1 = max(c1w1, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
                c1w0 = min(c1w0, 6.0f);
                c1w1 = min(c1w1, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst[1 * dst_c_stride + 0] = c1w0;
            l_dst[1 * dst_c_stride + 1] = c1w1;
            l_dst += 2;
        }
        if (w & 1) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c1w0 = l_dst_trans[0 * CH_DT_BLK() + 1];

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c1w0 += l_sum[1 * dst_c_stride + 0];
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c1w0 = max(c1w0, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c1w0 = min(c1w0, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[1 * dst_c_stride + 0] = c1w0;
        }
    } else {
        while (w >= unroll_w) {
            w -= unroll_w;
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            float c0w2 = l_dst_trans[2 * CH_DT_BLK() + 0];
            float c0w3 = l_dst_trans[3 * CH_DT_BLK() + 0];
            l_dst_trans += unroll_w * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                c0w2 += l_sum[0 * dst_c_stride + 2];
                c0w3 += l_sum[0 * dst_c_stride + 3];
                l_sum += unroll_w;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
                c0w2 = max(c0w2, 0.0f);
                c0w3 = max(c0w3, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
                c0w2 = min(c0w2, 6.0f);
                c0w3 = min(c0w3, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst[0 * dst_c_stride + 2] = c0w2;
            l_dst[0 * dst_c_stride + 3] = c0w3;
            l_dst += unroll_w;
        }
        if (w & 2) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];
            float c0w1 = l_dst_trans[1 * CH_DT_BLK() + 0];
            l_dst_trans += 2 * CH_DT_BLK();

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
                c0w1 += l_sum[0 * dst_c_stride + 1];
                l_sum += 2;
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
                c0w1 = max(c0w1, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
                c0w1 = min(c0w1, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
            l_dst[0 * dst_c_stride + 1] = c0w1;
            l_dst += 2;
        }
        if (w & 1) {
            float c0w0 = l_dst_trans[0 * CH_DT_BLK() + 0];

            if (sum) {
                c0w0 += l_sum[0 * dst_c_stride + 0];
            }

            if (relu >= 1) {
                c0w0 = max(c0w0, 0.0f);
            }

            if (relu == 6) {
                c0w0 = min(c0w0, 6.0f);
            }

            l_dst[0 * dst_c_stride + 0] = c0w0;
        }
    }
}

void conv2d_depthwise_fp32_sse_dst_trans_simple(
    const float *dst_trans,
    const float *sum_src,
    int64_t width,
    int64_t channels,
    int64_t dst_c_stride,
    float *dst)
{
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t w = 0; w < width; ++w) {
            dst[c * dst_c_stride + w] = dst_trans[w * CH_DT_BLK() + c];
        }
    }
}

ppl::common::RetCode conv2d_depthwise_fp32_sse_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t src_c = src_shape_->GetDim(1);
    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_c = dst_shape_->GetDim(1);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);
    const int64_t padded_src_w = (src_w + 2 * cp.pad_w);

    const int64_t ext_kernel_h = (cp.kernel_h - 1) * cp.dilation_h + 1;

    const int64_t src_b_stride = int64_t(src_c) * src_h * src_w;
    const int64_t src_c_stride = int64_t(src_h) * src_w;
    const int64_t src_trans_h_stride = int64_t(padded_src_w) * CH_DT_BLK();
    const int64_t src_trans_sw_stride = cp.stride_w * CH_DT_BLK();
    const int64_t dst_b_stride = int64_t(dst_c) * dst_h * dst_w;
    const int64_t dst_c_stride = int64_t(dst_h) * dst_w;

    const int64_t src_trans_len = src_h * padded_src_w * CH_DT_BLK();
    const int64_t dst_buf_len   = dst_w * CH_DT_BLK();
    const int64_t thread_buf_len = src_trans_len + dst_buf_len;

    const bool with_sum = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        const int64_t sum_src_c = sum_src_shape_->GetDim(1);
        sum_src_b_stride = int64_t(sum_src_c) * dst_h * dst_w;
    }

    int64_t share_param[SHAR_PARAM_LEN()];
    share_param[SRC_SW_STRIDE_IDX()] = src_trans_sw_stride;
    share_param[SRC_DH_STRIDE_IDX()] = cp.dilation_h * src_trans_h_stride;
    share_param[SRC_DW_STRIDE_IDX()] = cp.dilation_w * CH_DT_BLK();
    share_param[KW_IDX()] = cp.kernel_w;
    const int32_t stride_w_sel = cp.stride_w > 2 ? 0: cp.stride_w;

    auto dst_trans_func = conv2d_depthwise_fp32_sse_dst_trans<0, 0>;
    if (with_sum) {
        if (with_relu) dst_trans_func = conv2d_depthwise_fp32_sse_dst_trans<1, 1>;
        else if (with_relu6) dst_trans_func = conv2d_depthwise_fp32_sse_dst_trans<6, 1>;
        else dst_trans_func = conv2d_depthwise_fp32_sse_dst_trans<0, 1>;
    } else {
        if (with_relu) dst_trans_func = conv2d_depthwise_fp32_sse_dst_trans<1, 0>;
        else if (with_relu6) dst_trans_func = conv2d_depthwise_fp32_sse_dst_trans<6, 0>;
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t bc = 0; bc < batch * sp.padded_ch; bc += CH_DT_BLK()) {
        int64_t private_param[PRIV_PARAM_LEN()];
        const int64_t b           = bc / sp.padded_ch;
        const int64_t c           = bc % sp.padded_ch;
        const int64_t c_eff       = min<int64_t>(cp.group - c, CH_DT_BLK());
        const float *base_src     = src_ + b * src_b_stride + c * src_c_stride;
        const float *base_sum_src = sum_src_ + b * sum_src_b_stride + c * dst_c_stride;
        float *base_dst           = dst_ + b * dst_b_stride + c * dst_c_stride;

        PICK_PARAM(const float*, private_param, FLT_IDX()) = cvt_filter_ + c * cp.kernel_h * cp.kernel_w;
        PICK_PARAM(const float*, private_param, BIAS_IDX()) = cvt_bias_ + c;

        float *src_trans = reinterpret_cast<float*>(temp_buffer_) + PPL_OMP_THREAD_ID() * thread_buf_len;
        float *dst_buf   = src_trans + src_trans_len;

        { // transpose
            float *l_src_trans = src_trans;
            for (int64_t ih = 0; ih < src_h; ++ih) {
                memset32_sse(l_src_trans, 0, cp.pad_w * CH_DT_BLK());
                l_src_trans += cp.pad_w * CH_DT_BLK();
                const int64_t unroll_w = 4;
                int64_t iw = src_w;
                if (c_eff == CH_DT_BLK()) {
                    while (iw >= unroll_w * 2) {
                        iw -= 2 * unroll_w;
                        transpose_4x4_fp32_sse(base_src + 0 * unroll_w, src_c_stride, CH_DT_BLK(), l_src_trans + 0 * unroll_w * CH_DT_BLK());
                        transpose_4x4_fp32_sse(base_src + 1 * unroll_w, src_c_stride, CH_DT_BLK(), l_src_trans + 1 * unroll_w * CH_DT_BLK());
                        base_src += 2 * unroll_w;
                        l_src_trans += 2 * unroll_w * CH_DT_BLK();
                    }
                    if (iw & 4) {
                        transpose_4x4_fp32_sse(base_src + 0 * unroll_w, src_c_stride, CH_DT_BLK(), l_src_trans + 0 * unroll_w * CH_DT_BLK());
                        iw -= 1 * unroll_w;
                        base_src += 1 * unroll_w;
                        l_src_trans += 1 * unroll_w * CH_DT_BLK();
                    }
                    if (iw & 2) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[2 + 0 * CH_DT_BLK()] = base_src[2 * src_c_stride + 0];
                        l_src_trans[3 + 0 * CH_DT_BLK()] = base_src[3 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        l_src_trans[1 + 1 * CH_DT_BLK()] = base_src[1 * src_c_stride + 1];
                        l_src_trans[2 + 1 * CH_DT_BLK()] = base_src[2 * src_c_stride + 1];
                        l_src_trans[3 + 1 * CH_DT_BLK()] = base_src[3 * src_c_stride + 1];
                        base_src += 2;
                        l_src_trans += 2 * CH_DT_BLK();
                    }
                    if (iw & 1) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[2 + 0 * CH_DT_BLK()] = base_src[2 * src_c_stride + 0];
                        l_src_trans[3 + 0 * CH_DT_BLK()] = base_src[3 * src_c_stride + 0];
                        base_src += 1;
                        l_src_trans += 1 * CH_DT_BLK();
                    }
                } else if (c_eff == 3) {
                    while (iw >= unroll_w) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[2 + 0 * CH_DT_BLK()] = base_src[2 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        l_src_trans[1 + 1 * CH_DT_BLK()] = base_src[1 * src_c_stride + 1];
                        l_src_trans[2 + 1 * CH_DT_BLK()] = base_src[2 * src_c_stride + 1];
                        l_src_trans[0 + 2 * CH_DT_BLK()] = base_src[0 * src_c_stride + 2];
                        l_src_trans[1 + 2 * CH_DT_BLK()] = base_src[1 * src_c_stride + 2];
                        l_src_trans[2 + 2 * CH_DT_BLK()] = base_src[2 * src_c_stride + 2];
                        l_src_trans[0 + 3 * CH_DT_BLK()] = base_src[0 * src_c_stride + 3];
                        l_src_trans[1 + 3 * CH_DT_BLK()] = base_src[1 * src_c_stride + 3];
                        l_src_trans[2 + 3 * CH_DT_BLK()] = base_src[2 * src_c_stride + 3];
                        iw -= unroll_w;
                        base_src += unroll_w;
                        l_src_trans += unroll_w * CH_DT_BLK();
                    }
                    if (iw & 2) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[2 + 0 * CH_DT_BLK()] = base_src[2 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        l_src_trans[1 + 1 * CH_DT_BLK()] = base_src[1 * src_c_stride + 1];
                        l_src_trans[2 + 1 * CH_DT_BLK()] = base_src[2 * src_c_stride + 1];
                        base_src += 2;
                        l_src_trans += 2 * CH_DT_BLK();
                    }
                    if (iw & 1) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[2 + 0 * CH_DT_BLK()] = base_src[2 * src_c_stride + 0];
                        base_src += 1;
                        l_src_trans += 1 * CH_DT_BLK();
                    }
                } else if (c_eff == 2) {
                    while (iw >= unroll_w) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        l_src_trans[1 + 1 * CH_DT_BLK()] = base_src[1 * src_c_stride + 1];
                        l_src_trans[0 + 2 * CH_DT_BLK()] = base_src[0 * src_c_stride + 2];
                        l_src_trans[1 + 2 * CH_DT_BLK()] = base_src[1 * src_c_stride + 2];
                        l_src_trans[0 + 3 * CH_DT_BLK()] = base_src[0 * src_c_stride + 3];
                        l_src_trans[1 + 3 * CH_DT_BLK()] = base_src[1 * src_c_stride + 3];
                        iw -= unroll_w;
                        base_src += unroll_w;
                        l_src_trans += unroll_w * CH_DT_BLK();
                    }
                    if (iw & 2) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        l_src_trans[1 + 1 * CH_DT_BLK()] = base_src[1 * src_c_stride + 1];
                        base_src += 2;
                        l_src_trans += 2 * CH_DT_BLK();
                    }
                    if (iw & 1) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[1 + 0 * CH_DT_BLK()] = base_src[1 * src_c_stride + 0];
                        base_src += 1;
                        l_src_trans += 1 * CH_DT_BLK();
                    }
                } else {
                    while (iw >= unroll_w) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        l_src_trans[0 + 2 * CH_DT_BLK()] = base_src[0 * src_c_stride + 2];
                        l_src_trans[0 + 3 * CH_DT_BLK()] = base_src[0 * src_c_stride + 3];
                        iw -= unroll_w;
                        base_src += unroll_w;
                        l_src_trans += unroll_w * CH_DT_BLK();
                    }
                    if (iw & 2) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        l_src_trans[0 + 1 * CH_DT_BLK()] = base_src[0 * src_c_stride + 1];
                        base_src += 2;
                        l_src_trans += 2 * CH_DT_BLK();
                    }
                    if (iw & 1) {
                        l_src_trans[0 + 0 * CH_DT_BLK()] = base_src[0 * src_c_stride + 0];
                        base_src += 1;
                        l_src_trans += 1 * CH_DT_BLK();
                    }
                }
                memset32_sse(l_src_trans, 0, cp.pad_w * CH_DT_BLK());
                l_src_trans += cp.pad_w * CH_DT_BLK();
            }
        }

        const int64_t ow_unroll_body = round(dst_w, sp.ow_kr_blk);
        const int64_t ow_unroll_tail = dst_w - ow_unroll_body;

        for (int64_t oh = 0; oh < dst_h; ++oh) {
            const int64_t ih = oh * cp.stride_h - cp.pad_h;
            if (cp.dilation_h == 1) {
                private_param[KH_START_IDX()] = min<int64_t>(max<int64_t>(0 - ih, 0), cp.kernel_h - 1);
                private_param[KH_END_IDX()]   = max<int64_t>(min<int64_t>(src_h - ih, cp.kernel_h), 0);
            } else {
                private_param[KH_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - ih, 0), ext_kernel_h - 1), cp.dilation_h);
                private_param[KH_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_h - ih, ext_kernel_h), 0), cp.dilation_h);
            }

            PICK_PARAM(const float*, private_param, SRC_IDX()) = src_trans + ih * src_trans_h_stride;
            PICK_PARAM(float*, private_param, DST_IDX())       = dst_buf;

            if (ow_unroll_body) {
                private_param[OW_IDX()] = ow_unroll_body;
                conv2d_depthwise_kernel_fp32_sse_table[stride_w_sel][sp.ow_kr_blk - 1](share_param, private_param);
            }
            if (ow_unroll_tail) {
                private_param[OW_IDX()] = ow_unroll_tail;
                conv2d_depthwise_kernel_fp32_sse_table[stride_w_sel][ow_unroll_tail - 1](share_param, private_param);
            }

            dst_trans_func(dst_buf, base_sum_src, dst_w, c_eff, dst_c_stride, base_dst);
            base_sum_src += dst_w;
            base_dst += dst_w;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_depthwise_fp32_sse_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t channels  = param_.group;
    const int64_t padded_ch = round_up(channels, CH_DT_BLK());

    cvt_bias_size_ = padded_ch;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }
    memcpy(cvt_bias_, bias, channels * sizeof(float));
    memset(cvt_bias_ + channels, 0, (padded_ch - channels) * sizeof(float));

    cvt_filter_size_ = padded_ch * param_.kernel_h * param_.kernel_w;
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    ppl::nn::TensorShape filter_shape;
    filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    filter_shape.Reshape({1, channels, param_.kernel_h, param_.kernel_w});

    return reorder_ndarray_n4cx_fp32(&filter_shape, filter, cvt_filter_);
}

bool conv2d_depthwise_fp32_sse_manager::is_supported()
{
    return param_.is_depthwise();
}

conv2d_fp32_executor *conv2d_depthwise_fp32_sse_manager::gen_executor()
{
    return new conv2d_depthwise_fp32_sse_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
