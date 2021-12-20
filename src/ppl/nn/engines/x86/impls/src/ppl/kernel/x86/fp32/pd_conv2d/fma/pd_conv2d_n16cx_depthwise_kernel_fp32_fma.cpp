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

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/pd_conv2d/fma/pd_conv2d_n16cx_depthwise_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t u_w>
void pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel(int64_t *param)
{
#define KW_COMPUTE_STEP() do {\
    ymm14 = _mm256_loadu_ps(flt + 0 * CH_REG_ELTS);\
    ymm15 = _mm256_loadu_ps(flt + 1 * CH_REG_ELTS);\
    if (u_w > 0) {\
        ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 0 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm0);\
        ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 0 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm1);\
    }\
    if (u_w > 1) {\
        ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 1 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm2);\
        ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 1 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm3);\
    }\
    if (u_w > 2) {\
        ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 2 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm4);\
        ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 2 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm5);\
    }\
    if (u_w > 3) {\
        ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 3 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm6);\
        ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 3 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm7);\
    }\
    if (u_w > 4) {\
        ymm8 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 4 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm8);\
        ymm9 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 4 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm9);\
    }\
    if (u_w > 5) {\
        ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 5 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm10);\
        ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 5 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm11);\
    }\
    if (u_w > 6) {\
        ymm12 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 6 * src_sw_stride + 0 * CH_REG_ELTS), ymm14, ymm12);\
        ymm13 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 6 * src_sw_stride + 1 * CH_REG_ELTS), ymm15, ymm13);\
    }\
    flt += CH_DATA_BLK;\
    src += CH_DATA_BLK;\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    const int64_t CH_DATA_BLK = pd_conv2d_n16cx_depthwise_kernel_fp32_fma::config::CH_DATA_BLK;
    const int64_t CH_REG_ELTS = pd_conv2d_n16cx_depthwise_kernel_fp32_fma::config::CH_REG_ELTS;

    array_param_helper ker_p(param);

    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DATA_BLK : 
                                  ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::SRC_SW_STRIDE_IDX);
    const int64_t kernel_flags  = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::FLAGS_IDX);
    const int64_t kernel_w      = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::KW_IDX);

    const int64_t kh_start = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::KH_START_IDX);
    const int64_t kh_end   = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::KH_END_IDX);

    const int64_t src_uw_stride = u_w * src_sw_stride;
    const int64_t flt_offset    = kh_start * kernel_w * CH_DATA_BLK;

    const float **src_kh_list = ker_p.pick<const float**>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::SRC_PTR_KH_LIST_IDX);
    float *dst                = ker_p.pick<float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::DST_PTR_IDX);
    int64_t dst_w             = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::DST_WIDTH_IDX);
    do {
        const float* bias = ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::BIAS_PTR_IDX);
        if (u_w > 0)  {
            ymm0 = _mm256_loadu_ps(bias + 0 * CH_REG_ELTS);
            ymm1 = _mm256_loadu_ps(bias + 1 * CH_REG_ELTS);
        }
        if (u_w > 1)  {
            ymm2 = ymm0;
            ymm3 = ymm1;
        }
        if (u_w > 2)  {
            ymm4 = ymm0;
            ymm5 = ymm1;
        }
        if (u_w > 3)  {
            ymm6 = ymm0;
            ymm7 = ymm1;
        }
        if (u_w > 4)  {
            ymm8 = ymm0;
            ymm9 = ymm1;
        }
        if (u_w > 5)  {
            ymm10 = ymm0;
            ymm11 = ymm1;
        }
        if (u_w > 6)  {
            ymm12 = ymm0;
            ymm13 = ymm1;
        }

        const float *flt = ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::FLT_PTR_IDX) + flt_offset;
        if (kernel_w == 3) {
            for (int32_t kh = kh_start; kh < kh_end; ++kh) {
                const float *src = src_kh_list[kh];
                src_kh_list[kh] = src + src_uw_stride;
                KW_COMPUTE_STEP();
                KW_COMPUTE_STEP();
                KW_COMPUTE_STEP();
            }
        } else {
            for (int32_t kh = kh_start; kh < kh_end; ++kh) {
                const float *src = src_kh_list[kh];
                src_kh_list[kh] = src + src_uw_stride;
                for (int32_t kw = 0; kw < kernel_w; ++kw) {
                    KW_COMPUTE_STEP();
                }
            }
        }

        if (kernel_flags & (pd_conv2d_n16cx_depthwise_kernel_fp32_fma::flag::RELU | pd_conv2d_n16cx_depthwise_kernel_fp32_fma::flag::RELU6)) {
            ymm14 = _mm256_setzero_ps();
            if (u_w > 0) {
                ymm0 = _mm256_max_ps(ymm0, ymm14);
                ymm1 = _mm256_max_ps(ymm1, ymm14);
            }
            if (u_w > 1) {
                ymm2 = _mm256_max_ps(ymm2, ymm14);
                ymm3 = _mm256_max_ps(ymm3, ymm14);
            }
            if (u_w > 2) {
                ymm4 = _mm256_max_ps(ymm4, ymm14);
                ymm5 = _mm256_max_ps(ymm5, ymm14);
            }
            if (u_w > 3) {
                ymm6 = _mm256_max_ps(ymm6, ymm14);
                ymm7 = _mm256_max_ps(ymm7, ymm14);
            }
            if (u_w > 4) {
                ymm8 = _mm256_max_ps(ymm8, ymm14);
                ymm9 = _mm256_max_ps(ymm9, ymm14);
            }
            if (u_w > 5) {
                ymm10 = _mm256_max_ps(ymm10, ymm14);
                ymm11 = _mm256_max_ps(ymm11, ymm14);
            }
            if (u_w > 6) {
                ymm12 = _mm256_max_ps(ymm12, ymm14);
                ymm13 = _mm256_max_ps(ymm13, ymm14);
            }
        }
        if (kernel_flags & pd_conv2d_n16cx_depthwise_kernel_fp32_fma::flag::RELU6) {
            ymm15 = _mm256_set1_ps(6.0f);
            if (u_w > 0) {
                ymm0 = _mm256_min_ps(ymm0, ymm15);
                ymm1 = _mm256_min_ps(ymm1, ymm15);
            }
            if (u_w > 1) {
                ymm2 = _mm256_min_ps(ymm2, ymm15);
                ymm3 = _mm256_min_ps(ymm3, ymm15);
            }
            if (u_w > 2) {
                ymm4 = _mm256_min_ps(ymm4, ymm15);
                ymm5 = _mm256_min_ps(ymm5, ymm15);
            }
            if (u_w > 3) {
                ymm6 = _mm256_min_ps(ymm6, ymm15);
                ymm7 = _mm256_min_ps(ymm7, ymm15);
            }
            if (u_w > 4) {
                ymm8 = _mm256_min_ps(ymm8, ymm15);
                ymm9 = _mm256_min_ps(ymm9, ymm15);
            }
            if (u_w > 5) {
                ymm10 = _mm256_min_ps(ymm10, ymm15);
                ymm11 = _mm256_min_ps(ymm11, ymm15);
            }
            if (u_w > 6) {
                ymm12 = _mm256_min_ps(ymm12, ymm15);
                ymm13 = _mm256_min_ps(ymm13, ymm15);
            }
        }
        if (nt_store) {
            if (u_w > 0) {
                _mm256_stream_ps(dst + 0 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm0);
                _mm256_stream_ps(dst + 0 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm1);
            }
            if (u_w > 1) {
                _mm256_stream_ps(dst + 1 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm2);
                _mm256_stream_ps(dst + 1 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm3);
            }
            if (u_w > 2) {
                _mm256_stream_ps(dst + 2 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm4);
                _mm256_stream_ps(dst + 2 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm5);
            }
            if (u_w > 3) {
                _mm256_stream_ps(dst + 3 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm6);
                _mm256_stream_ps(dst + 3 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm7);
            }
            if (u_w > 4) {
                _mm256_stream_ps(dst + 4 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm8);
                _mm256_stream_ps(dst + 4 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm9);
            }
            if (u_w > 5) {
                _mm256_stream_ps(dst + 5 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm10);
                _mm256_stream_ps(dst + 5 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm11);
            }
            if (u_w > 6) {
                _mm256_stream_ps(dst + 6 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm12);
                _mm256_stream_ps(dst + 6 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm13);
            }
        } else {
            if (u_w > 0) {
                _mm256_storeu_ps(dst + 0 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm0);
                _mm256_storeu_ps(dst + 0 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm1);
            }
            if (u_w > 1) {
                _mm256_storeu_ps(dst + 1 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm2);
                _mm256_storeu_ps(dst + 1 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm3);
            }
            if (u_w > 2) {
                _mm256_storeu_ps(dst + 2 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm4);
                _mm256_storeu_ps(dst + 2 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm5);
            }
            if (u_w > 3) {
                _mm256_storeu_ps(dst + 3 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm6);
                _mm256_storeu_ps(dst + 3 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm7);
            }
            if (u_w > 4) {
                _mm256_storeu_ps(dst + 4 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm8);
                _mm256_storeu_ps(dst + 4 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm9);
            }
            if (u_w > 5) {
                _mm256_storeu_ps(dst + 5 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm10);
                _mm256_storeu_ps(dst + 5 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm11);
            }
            if (u_w > 6) {
                _mm256_storeu_ps(dst + 6 * CH_DATA_BLK + 0 * CH_REG_ELTS, ymm12);
                _mm256_storeu_ps(dst + 6 * CH_DATA_BLK + 1 * CH_REG_ELTS, ymm13);
            }
        }
        dst += u_w * CH_DATA_BLK;
        dst_w -= u_w;
    } while (dst_w > 0);
    ker_p.pick<float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::DST_PTR_IDX) = dst;
#undef KW_COMPUTE_STEP
}

#define PD_CONV2D_DW_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
{\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 1>,\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 2>,\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 3>,\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 4>,\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 5>,\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 6>,\
    pd_conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 7>,\
}

const pd_conv2d_n16cx_depthwise_kernel_fp32_fma::func_t
    pd_conv2d_n16cx_depthwise_kernel_fp32_fma::table_[config::NT_STORE_OPT][config::SPEC_STRIDE_W_OPT][config::MAX_W_REGS] =
{
    {
        PD_CONV2D_DW_KERNEL_TABLE_BLK(false, 0),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(false, 1),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(false, 2),
    },
    {
        PD_CONV2D_DW_KERNEL_TABLE_BLK(true, 0),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(true, 1),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(true, 2),
    },
};

}}};
