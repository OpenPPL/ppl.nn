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

#include "ppl/kernel/x86/fp32/gemm/fma/gemm_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

template<int64_t u_m, int64_t u_n>
void gemm_fp32_fma_kernel(int64_t *param) {

#define K_COMPUTE_STEP(K) do {\
    if (u_nr > 0) ymm13 = _mm256_loadu_ps(kpacked_b + 0 * ldpacked_b + K * n_reg_elts);\
    if (u_nr > 1) ymm14 = _mm256_loadu_ps(kpacked_b + 1 * ldpacked_b + K * n_reg_elts);\
    if (u_nr > 2) ymm15 = _mm256_loadu_ps(kpacked_b + 2 * ldpacked_b + K * n_reg_elts);\
    if (u_m > 0) {\
        ymm12 = _mm256_set1_ps(ka_m0[0 * lda + K]);\
        if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm13, ymm12, ymm0);\
        if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm14, ymm12, ymm1);\
        if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm12, ymm2);\
    }\
    if (u_m > 1) {\
        ymm12 = _mm256_set1_ps(ka_m0[1 * lda + K]);\
        if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm13, ymm12, ymm3);\
        if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm14, ymm12, ymm4);\
        if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm12, ymm5);\
    }\
    if (u_m > 2) {\
        ymm12 = _mm256_set1_ps(ka_m2[0 * lda + K]);\
        if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm13, ymm12, ymm6);\
        if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm14, ymm12, ymm7);\
        if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm12, ymm8);\
    }\
    if (u_m > 3) {\
        ymm12 = _mm256_set1_ps(ka_m2[1 * lda + K]);\
        if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm13, ymm12, ymm9);\
        if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm14, ymm12, ymm10);\
        if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm12, ymm11);\
    }\
} while (0)

    array_param_helper kp(param);
    const int64_t n_reg_elts = gemm_kernel_fp32_fma::config::n_reg_elts;
    const int64_t u_nr = div_up(u_n, n_reg_elts);
    const int64_t u_k = 8;

    const gemm_kernel_fp32_fma::flag_t flags = kp.pick<const gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::flags_idx);
    const float *packed_b_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx);
    float *c_ptr = kp.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx);
    const int64_t ldpacked_b = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::ldpacked_b_idx);
    int64_t n = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx);
    do {
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
        __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
        { // session - initialize
            if (flags & gemm_kernel_fp32_fma::flag::load_c) {
                const int64_t ldc = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::ldc_idx);
                const float *l_c = c_ptr;
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_loadu_ps(l_c + 0 * n_reg_elts);
                    if (u_nr > 1) ymm1 = _mm256_loadu_ps(l_c + 1 * n_reg_elts);
                    if (u_nr > 2) ymm2 = _mm256_loadu_ps(l_c + 2 * n_reg_elts);
                }
                if (u_m > 1) {
                    l_c += ldc;
                    if (u_nr > 0) ymm3 = _mm256_loadu_ps(l_c + 0 * n_reg_elts);
                    if (u_nr > 1) ymm4 = _mm256_loadu_ps(l_c + 1 * n_reg_elts);
                    if (u_nr > 2) ymm5 = _mm256_loadu_ps(l_c + 2 * n_reg_elts);
                }
                if (u_m > 2) {
                    l_c += ldc;
                    if (u_nr > 0) ymm6 = _mm256_loadu_ps(l_c + 0 * n_reg_elts);
                    if (u_nr > 1) ymm7 = _mm256_loadu_ps(l_c + 1 * n_reg_elts);
                    if (u_nr > 2) ymm8 = _mm256_loadu_ps(l_c + 2 * n_reg_elts);
                }
                if (u_m > 3) {
                    l_c += ldc;
                    if (u_nr > 0) ymm9 = _mm256_loadu_ps(l_c + 0 * n_reg_elts);
                    if (u_nr > 1) ymm10 = _mm256_loadu_ps(l_c + 1 * n_reg_elts);
                    if (u_nr > 2) ymm11 = _mm256_loadu_ps(l_c + 2 * n_reg_elts);
                }
            } else {
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_setzero_ps();
                    if (u_nr > 1) ymm1 = _mm256_setzero_ps();
                    if (u_nr > 2) ymm2 = _mm256_setzero_ps();
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_setzero_ps();
                    if (u_nr > 1) ymm4 = _mm256_setzero_ps();
                    if (u_nr > 2) ymm5 = _mm256_setzero_ps();
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_setzero_ps();
                    if (u_nr > 1) ymm7 = _mm256_setzero_ps();
                    if (u_nr > 2) ymm8 = _mm256_setzero_ps();
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_setzero_ps();
                    if (u_nr > 1) ymm10 = _mm256_setzero_ps();
                    if (u_nr > 2) ymm11 = _mm256_setzero_ps();
                }
            }
        }

        { // session - compute
            int64_t k = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::k_idx);
            const int64_t lda = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::lda_idx);
            const float *kpacked_b = packed_b_ptr;
            const float *ka_m0;
            const float *ka_m2;
            if (u_m > 0) ka_m0 = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::a_ptr_idx);
            if (u_m > 2) ka_m2 = ka_m0 + 2 * lda;
            while (k >= u_k) {
                k -= u_k;
                K_COMPUTE_STEP(0);
                K_COMPUTE_STEP(1);
                K_COMPUTE_STEP(2);
                K_COMPUTE_STEP(3);
                K_COMPUTE_STEP(4);
                K_COMPUTE_STEP(5);
                K_COMPUTE_STEP(6);
                K_COMPUTE_STEP(7);
                if (u_m > 0) ka_m0 += u_k;
                if (u_m > 2) ka_m2 += u_k;
                kpacked_b += u_k * n_reg_elts;
            }
            while (k > 0) {
                --k;
                K_COMPUTE_STEP(0);
                if (u_m > 0) ka_m0 += 1;
                if (u_m > 2) ka_m2 += 1;
                kpacked_b += n_reg_elts;
            }
        }
        
        { // session - finalize
            const float alpha = kp.pick<const float>(gemm_kernel_fp32_fma::param_def::alpha_idx);
            if (alpha != 1.0f) {
                ymm12 = _mm256_set1_ps(alpha);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_mul_ps(ymm12, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_mul_ps(ymm12, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_mul_ps(ymm12, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_mul_ps(ymm12, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_mul_ps(ymm12, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_mul_ps(ymm12, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_mul_ps(ymm12, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_mul_ps(ymm12, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_mul_ps(ymm12, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_mul_ps(ymm12, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_mul_ps(ymm12, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_mul_ps(ymm12, ymm11);
                }
            }

            if (flags & (gemm_kernel_fp32_fma::flag::relu | gemm_kernel_fp32_fma::flag::relu6)) {
                ymm13 = _mm256_setzero_ps();
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_max_ps(ymm13, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_max_ps(ymm13, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_max_ps(ymm13, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_max_ps(ymm13, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_max_ps(ymm13, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_max_ps(ymm13, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_max_ps(ymm13, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_max_ps(ymm13, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_max_ps(ymm13, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_max_ps(ymm13, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_max_ps(ymm13, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_max_ps(ymm13, ymm11);
                }
            }

            if (flags & gemm_kernel_fp32_fma::flag::relu6) {
                ymm12 = _mm256_set1_ps(6.0f);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_min_ps(ymm12, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_min_ps(ymm12, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_min_ps(ymm12, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_min_ps(ymm12, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_min_ps(ymm12, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_min_ps(ymm12, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_min_ps(ymm12, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_min_ps(ymm12, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_min_ps(ymm12, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_min_ps(ymm12, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_min_ps(ymm12, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_min_ps(ymm12, ymm11);
                }
            }

            const int64_t ldc = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::ldc_idx);
            float *l_c = c_ptr;
            if (u_m > 0) {
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm0);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm1);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm2);
            }
            if (u_m > 1) {
                l_c += ldc;
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm3);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm4);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm5);
            }
            if (u_m > 2) {
                l_c += ldc;
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm6);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm7);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm8);
            }
            if (u_m > 3) {
                l_c += ldc;
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm9);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm10);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm11);
            }
        }
        { // next n block
            packed_b_ptr += u_nr * ldpacked_b;
            c_ptr += u_n;
            n -= u_n;
        }
    } while (n > 0);
#undef K_COMPUTE_STEP
}

const gemm_kernel_fp32_fma::func_t gemm_kernel_fp32_fma::table_[gemm_kernel_fp32_fma::config::max_n_regs][gemm_kernel_fp32_fma::config::max_m_regs] =
{
    {
        gemm_fp32_fma_kernel<1, 1 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<2, 1 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<3, 1 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<4, 1 * gemm_kernel_fp32_fma::config::n_reg_elts>,
    },
    {
        gemm_fp32_fma_kernel<1, 2 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<2, 2 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<3, 2 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<4, 2 * gemm_kernel_fp32_fma::config::n_reg_elts>,
    },
    {
        gemm_fp32_fma_kernel<1, 3 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<2, 3 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<3, 3 * gemm_kernel_fp32_fma::config::n_reg_elts>,
        gemm_fp32_fma_kernel<4, 3 * gemm_kernel_fp32_fma::config::n_reg_elts>,
    },
};

}}}; // namespace ppl::kernel::x86

