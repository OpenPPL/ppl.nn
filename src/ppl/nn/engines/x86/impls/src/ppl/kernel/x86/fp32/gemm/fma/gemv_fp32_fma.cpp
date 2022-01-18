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

#include <stdio.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

template<int64_t u_n>
void gemv_t_kernel_fp32_fma(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
    const gemm_v_type_t typeV,
    const gemm_m_type_t typeH,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const gemm_post_t post,
    float *C)
{
    const float *lB = B;
    const float *lV = V;
    const float *lH = H;
    float *lC = C;

    const int64_t K_REG_ELTS = 8;
    const int64_t u_k = K_REG_ELTS;
    const bool apply_beta = beta != 0.0f;
    for (int64_t n = 0; n < N; n += u_n) {
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4;
        float r0, r1, r2, r3;

        r0 = 0.0f;
        r1 = 0.0f;
        r2 = 0.0f;
        r3 = 0.0f;

        const float *rA = A;
        const float *rB0;
        const float *rB2;
        if (u_n > 0) rB0 = lB + 0 * ldb;
        if (u_n > 2) rB2 = lB + 2 * ldb;

        int64_t k = 0;
        if (K >= u_k) {
            if (u_n > 0) ymm0 = _mm256_setzero_ps();
            if (u_n > 1) ymm1 = _mm256_setzero_ps();
            if (u_n > 2) ymm2 = _mm256_setzero_ps();
            if (u_n > 3) ymm3 = _mm256_setzero_ps();
            for (; k <= K - u_k; k += u_k) {
                ymm4 = _mm256_loadu_ps(rA);
                if (u_n > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 0 * ldb), ymm4, ymm0);
                if (u_n > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb), ymm4, ymm1);
                if (u_n > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb), ymm4, ymm2);
                if (u_n > 3) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb), ymm4, ymm3);
                rA += u_k;
                if (u_n > 0) rB0 += u_k;
                if (u_n > 2) rB2 += u_k;
            }
#define _MM256_RSUM_PS(YMM, SREG) do {\
    float v[8];\
    _mm256_storeu_ps(v, (YMM));\
    (SREG) = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));\
} while(0)
            if (u_n > 0) _MM256_RSUM_PS(ymm0, r0);
            if (u_n > 1) _MM256_RSUM_PS(ymm1, r1);
            if (u_n > 2) _MM256_RSUM_PS(ymm2, r2);
            if (u_n > 3) _MM256_RSUM_PS(ymm3, r3);
        }
        for (; k < K; ++k) {
            if (u_n > 0) r0 += rB0[0 * ldb] * rA[0];
            if (u_n > 1) r1 += rB0[1 * ldb] * rA[0];
            if (u_n > 2) r2 += rB2[0 * ldb] * rA[0];
            if (u_n > 3) r3 += rB2[1 * ldb] * rA[0];
            rA += 1;
            if (u_n > 0) rB0 += 1;
            if (u_n > 2) rB2 += 1;
        }
        if (u_n > 0) r0 *= alpha;
        if (u_n > 1) r1 *= alpha;
        if (u_n > 2) r2 *= alpha;
        if (u_n > 3) r3 *= alpha;

        if (apply_beta) {
            if (typeV == gemm_v_type::COL_VEC || typeV == gemm_v_type::SCALAR) {
                if (u_n > 0) r0 += beta * lV[0];
                if (u_n > 1) r1 += beta * lV[0];
                if (u_n > 2) r2 += beta * lV[0];
                if (u_n > 3) r3 += beta * lV[0];
            }
            if (typeV == gemm_v_type::ROW_VEC) {
                if (u_n > 0) r0 += beta * lV[0];
                if (u_n > 1) r1 += beta * lV[1];
                if (u_n > 2) r2 += beta * lV[2];
                if (u_n > 3) r3 += beta * lV[3];
                lV += u_n;
            }
            if (typeH == gemm_m_type::NOTRANS) {
                if (u_n > 0) r0 += beta * lH[0];
                if (u_n > 1) r1 += beta * lH[1];
                if (u_n > 2) r2 += beta * lH[2];
                if (u_n > 3) r3 += beta * lH[3];
                lH += u_n;
            }
        }

        if (post == gemm_post::RELU || post == gemm_post::RELU6) {
            if (u_n > 0) r0 = max(r0, 0.0f);
            if (u_n > 1) r1 = max(r1, 0.0f);
            if (u_n > 2) r2 = max(r2, 0.0f);
            if (u_n > 3) r3 = max(r3, 0.0f);
        }
        if (post == gemm_post::RELU6) {
            if (u_n > 0) r0 = min(r0, 6.0f);
            if (u_n > 1) r1 = min(r1, 6.0f);
            if (u_n > 2) r2 = min(r2, 6.0f);
            if (u_n > 3) r3 = min(r3, 6.0f);
        }

        if (u_n > 0) lC[0] = r0;
        if (u_n > 1) lC[1] = r1;
        if (u_n > 2) lC[2] = r2;
        if (u_n > 3) lC[3] = r3;

        lB += u_n * ldb;
        lC += u_n;
    }
}


template<int64_t u_k>
void gemv_n_kernel_fp32_fma(
    const float *A,
    const float *B,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const gemm_post_t post,
    float *C)
{
    const float *lB = B;
    const float *lA = A;
    __m256 ymm_v0 = _mm256_set1_ps(0.0f);
    __m256 ymm_v6 = _mm256_set1_ps(6.0f);

    const int64_t N_REG_ELTS = 8;
    const int64_t u_n = N_REG_ELTS * 4;
    for (int64_t k = 0; k < K; k += u_k) {
        __m256 ymm0, ymm1, ymm2, ymm3;
        __m256 ymm4, ymm5, ymm6, ymm7;

        const bool do_relu_max = k + u_k >= K && (post == gemm_post::RELU || post == gemm_post::RELU6);
        const bool do_relu_min = k + u_k >= K && post == gemm_post::RELU6;

        if (u_k > 0) ymm0 = _mm256_set1_ps(lA[0]);
        if (u_k > 1) ymm1 = _mm256_set1_ps(lA[1]);
        if (u_k > 2) ymm2 = _mm256_set1_ps(lA[2]);
        if (u_k > 3) ymm3 = _mm256_set1_ps(lA[3]);

        float *rC = C;
        const float *rB0;
        const float *rB2;
        if (u_k > 0) rB0 = lB + 0 * ldb;
        if (u_k > 2) rB2 = lB + 2 * ldb;
        int64_t n = N;
        while (n >= u_n) {
            if (u_k > 0) {
                ymm4 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 0 * N_REG_ELTS), ymm0);
                ymm5 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 1 * N_REG_ELTS), ymm0);
                ymm6 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 2 * N_REG_ELTS), ymm0);
                ymm7 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 3 * N_REG_ELTS), ymm0);
            }
            if (u_k > 1) {
                ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 0 * N_REG_ELTS), ymm1, ymm4);
                ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 1 * N_REG_ELTS), ymm1, ymm5);
                ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 2 * N_REG_ELTS), ymm1, ymm6);
                ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 3 * N_REG_ELTS), ymm1, ymm7);
            }
            if (u_k > 2) {
                ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 0 * N_REG_ELTS), ymm2, ymm4);
                ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 1 * N_REG_ELTS), ymm2, ymm5);
                ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 2 * N_REG_ELTS), ymm2, ymm6);
                ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 3 * N_REG_ELTS), ymm2, ymm7);
            }
            if (u_k > 3) {
                ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 0 * N_REG_ELTS), ymm3, ymm4);
                ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 1 * N_REG_ELTS), ymm3, ymm5);
                ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 2 * N_REG_ELTS), ymm3, ymm6);
                ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 3 * N_REG_ELTS), ymm3, ymm7);
            }
            ymm4 = _mm256_add_ps(_mm256_loadu_ps(rC + 0 * N_REG_ELTS), ymm4);
            ymm5 = _mm256_add_ps(_mm256_loadu_ps(rC + 1 * N_REG_ELTS), ymm5);
            ymm6 = _mm256_add_ps(_mm256_loadu_ps(rC + 2 * N_REG_ELTS), ymm6);
            ymm7 = _mm256_add_ps(_mm256_loadu_ps(rC + 3 * N_REG_ELTS), ymm7);
            if (do_relu_max) {
                ymm4 = _mm256_max_ps(ymm4, ymm_v0);
                ymm5 = _mm256_max_ps(ymm5, ymm_v0);
                ymm6 = _mm256_max_ps(ymm6, ymm_v0);
                ymm7 = _mm256_max_ps(ymm7, ymm_v0);
            }
            if (do_relu_min) {
                ymm4 = _mm256_min_ps(ymm4, ymm_v6);
                ymm5 = _mm256_min_ps(ymm5, ymm_v6);
                ymm6 = _mm256_min_ps(ymm6, ymm_v6);
                ymm7 = _mm256_min_ps(ymm7, ymm_v6);
            }
            _mm256_storeu_ps(rC + 0 * N_REG_ELTS, ymm4);
            _mm256_storeu_ps(rC + 1 * N_REG_ELTS, ymm5);
            _mm256_storeu_ps(rC + 2 * N_REG_ELTS, ymm6);
            _mm256_storeu_ps(rC + 3 * N_REG_ELTS, ymm7);
            rC += u_n;
            if (u_k > 0) rB0 += u_n;
            if (u_k > 2) rB2 += u_n;
            n -= u_n;
        }
        if (n >= N_REG_ELTS * 2) {
            if (u_k > 0) {
                ymm4 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 0 * N_REG_ELTS), ymm0);
                ymm5 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 1 * N_REG_ELTS), ymm0);
            }
            if (u_k > 1) {
                ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 0 * N_REG_ELTS), ymm1, ymm4);
                ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 1 * N_REG_ELTS), ymm1, ymm5);
            }
            if (u_k > 2) {
                ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 0 * N_REG_ELTS), ymm2, ymm4);
                ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 1 * N_REG_ELTS), ymm2, ymm5);
            }
            if (u_k > 3) {
                ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 0 * N_REG_ELTS), ymm3, ymm4);
                ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 1 * N_REG_ELTS), ymm3, ymm5);
            }
            ymm4 = _mm256_add_ps(_mm256_loadu_ps(rC + 0 * N_REG_ELTS), ymm4);
            ymm5 = _mm256_add_ps(_mm256_loadu_ps(rC + 1 * N_REG_ELTS), ymm5);
            if (do_relu_max) {
                ymm4 = _mm256_max_ps(ymm4, ymm_v0);
                ymm5 = _mm256_max_ps(ymm5, ymm_v0);
            }
            if (do_relu_min) {
                ymm4 = _mm256_min_ps(ymm4, ymm_v6);
                ymm5 = _mm256_min_ps(ymm5, ymm_v6);
            }
            _mm256_storeu_ps(rC + 0 * N_REG_ELTS, ymm4);
            _mm256_storeu_ps(rC + 1 * N_REG_ELTS, ymm5);
            rC += N_REG_ELTS * 2;
            if (u_k > 0) rB0 += N_REG_ELTS * 2;
            if (u_k > 2) rB2 += N_REG_ELTS * 2;
            n -= N_REG_ELTS * 2;
        }
        if (n >= N_REG_ELTS) {
            if (u_k > 0) ymm4 = _mm256_mul_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 0 * N_REG_ELTS), ymm0);
            if (u_k > 1) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 0 * N_REG_ELTS), ymm1, ymm4);
            if (u_k > 2) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 0 * N_REG_ELTS), ymm2, ymm4);
            if (u_k > 3) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 0 * N_REG_ELTS), ymm3, ymm4);
            ymm4 = _mm256_add_ps(_mm256_loadu_ps(rC + 0 * N_REG_ELTS), ymm4);
            if (do_relu_max) ymm4 = _mm256_max_ps(ymm4, ymm_v0);
            if (do_relu_min) ymm4 = _mm256_min_ps(ymm4, ymm_v6);
            _mm256_storeu_ps(rC + 0 * N_REG_ELTS, ymm4);
            rC += N_REG_ELTS;
            if (u_k > 0) rB0 += N_REG_ELTS;
            if (u_k > 2) rB2 += N_REG_ELTS;
            n -= N_REG_ELTS;
        }
        while (n > 0) {
            float r;
            if (u_k > 0) r = rB0[0 * ldb] * lA[0];
            if (u_k > 1) r += rB0[1 * ldb] * lA[1];
            if (u_k > 2) r += rB2[0 * ldb] * lA[2];
            if (u_k > 3) r += rB2[1 * ldb] * lA[3];
            r += rC[0];
            if (do_relu_max) r = max(r, 0.0f);
            if (do_relu_min) r = min(r, 6.0f);
            rC[0] = r;
            rC += 1;
            if (u_k > 0) rB0 += 1;
            if (u_k > 2) rB2 += 1;
            n -= 1;
        }

        lA += u_k;
        lB += u_k * ldb;
    }
}

template<gemm_v_type_t typeH, gemm_m_type_t typeV>
void gemv_fp32_apply_beta_avx(
    const float *V,
    const float *H,
    const int64_t N,
    const float beta,
    float *C)
{
    if ((typeV == gemm_v_type::EMPTY && typeH == gemm_m_type::EMPTY) || beta == 0.0f) {
        memset32_avx(C, 0, N);
        return;
    }

    const int64_t N_REG_ELTS = 8;
    const int64_t unroll_n = N_REG_ELTS * 2;
 
    __m256 ymm_v, ymm_beta;
    if (typeV == gemm_v_type::SCALAR || typeV == gemm_v_type::COL_VEC) ymm_v = _mm256_set1_ps(V[0]);
    ymm_beta = _mm256_set1_ps(beta);

    const float *lV = V;
    const float *lH = H;
    float *lC = C;
    int64_t n = N;
    while (n >= unroll_n) {
        __m256 ymm0, ymm1;
        if (typeV != gemm_v_type::EMPTY) {
            if (typeV == gemm_v_type::ROW_VEC) {
                ymm0 = _mm256_loadu_ps(lV + 0 * N_REG_ELTS);
                ymm1 = _mm256_loadu_ps(lV + 1 * N_REG_ELTS);
                lV += unroll_n;
            } else {
                ymm0 = ymm_v;
                ymm1 = ymm_v;
            }
        }
        if (typeH == gemm_m_type::NOTRANS) {
            if (typeV == gemm_v_type::EMPTY) {
                ymm0 = _mm256_loadu_ps(lH + 0 * N_REG_ELTS);
                ymm1 = _mm256_loadu_ps(lH + 1 * N_REG_ELTS);
            } else {
                ymm0 = _mm256_add_ps(_mm256_loadu_ps(lH + 0 * N_REG_ELTS), ymm0);
                ymm1 = _mm256_add_ps(_mm256_loadu_ps(lH + 1 * N_REG_ELTS), ymm1);
            }
            lH += unroll_n;
        }
        ymm0 = _mm256_mul_ps(ymm0, ymm_beta);
        ymm1 = _mm256_mul_ps(ymm1, ymm_beta);
        _mm256_storeu_ps(lC + 0 * N_REG_ELTS, ymm0);
        _mm256_storeu_ps(lC + 1 * N_REG_ELTS, ymm1);
        lC += unroll_n;
        n -= unroll_n;
    }
    while (n > 0) {
        float y = 0.0f;
        if (typeV != gemm_v_type::EMPTY) {
            if (typeV == gemm_v_type::SCALAR || typeV == gemm_v_type::COL_VEC) {
                y = V[0];
            }
            if (typeV == gemm_v_type::ROW_VEC) {
                y = lV[0];
                lV += 1;
            }
        }
        if (typeH == gemm_m_type::NOTRANS) {
            if (typeV == gemm_v_type::EMPTY) {
                y = lH[0];
            } else {
                y += lH[0];
            }
            lH += 1;
        }
        lC[0] = y * beta;
        lC += 1;
        n -= 1;
    }
}

ppl::common::RetCode gemv_operation_fp32_fma(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
    const gemm_v_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typeV,
    const gemm_m_type_t typeH,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const gemm_post_t post,
    float *C)
{
    if (typeA == gemm_v_type::COL_VEC || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeH != gemm_m_type::EMPTY && typeH != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool apply_aAB = alpha != 0.0f && typeA != gemm_v_type::EMPTY && typeB != gemm_m_type::EMPTY;
    const bool apply_bVpbH = beta != 0.0f && (typeV != gemm_v_type::EMPTY || typeH != gemm_m_type::EMPTY);

    if (!apply_aAB && !apply_bVpbH) {
        return ppl::common::RC_SUCCESS;
    }

    if (alpha == 0.0f && beta == 0.0f) {
        memset32_avx(C, 0, N);
        return ppl::common::RC_SUCCESS;
    }

    if (typeB == gemm_m_type::TRANS) {
        const int64_t MAX_U_N = 4;
        const int64_t n_body = round(N, MAX_U_N);
        const int64_t n_tail = N - n_body;

        const float *lB = B;
        const float *lV = V;
        const float *lH = H;
        float *lC = C;
        if (n_body) gemv_t_kernel_fp32_fma<MAX_U_N>(A, lB, lV, lH, typeV, typeH, n_body, K, ldb, alpha, beta, post, lC);
        
        lB += n_body * ldb;
        lC += n_body;

        if (typeV == gemm_v_type::ROW_VEC) {
            lV += n_body;
        }
        if (typeH == gemm_m_type::NOTRANS) {
            lH += n_body;
        }
        
        if (n_tail == 3) gemv_t_kernel_fp32_fma<3>(A, lB, lV, lH, typeV, typeH, n_tail, K, ldb, alpha, beta, post, lC);
        if (n_tail == 2) gemv_t_kernel_fp32_fma<2>(A, lB, lV, lH, typeV, typeH, n_tail, K, ldb, alpha, beta, post, lC);
        if (n_tail == 1) gemv_t_kernel_fp32_fma<1>(A, lB, lV, lH, typeV, typeH, n_tail, K, ldb, alpha, beta, post, lC);
    } else {
        auto apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
        if (typeH == gemm_m_type::NOTRANS) {
            if (typeV == gemm_v_type::EMPTY) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::EMPTY>;
            if (typeV == gemm_v_type::SCALAR) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::SCALAR>;
            if (typeV == gemm_v_type::COL_VEC) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::COL_VEC>;
            if (typeV == gemm_v_type::ROW_VEC) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::ROW_VEC>;
        } else {
            if (typeV == gemm_v_type::SCALAR) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::SCALAR>;
            if (typeV == gemm_v_type::COL_VEC) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::COL_VEC>;
            if (typeV == gemm_v_type::ROW_VEC) apply_beta_func = gemv_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::ROW_VEC>;
        }
        apply_beta_func(V, H, N, beta, C);

        const int64_t MAX_U_K = 4;
        const int64_t k_body = round(K, MAX_U_K);
        const int64_t k_tail = K - k_body;

        const float *lA = A;
        const float *lB = B;

        gemm_post_t l_post = gemm_post::NONE;
        if (!k_tail) l_post = post;

        if (k_body) gemv_n_kernel_fp32_fma<MAX_U_K>(lA, lB, N, k_body, ldb, l_post, C);

        lA += k_body;
        lB += k_body * ldb;

        l_post = post;

        if (k_tail == 3) gemv_n_kernel_fp32_fma<3>(lA, lB, N, k_tail, ldb, l_post, C);
        if (k_tail == 2) gemv_n_kernel_fp32_fma<2>(lA, lB, N, k_tail, ldb, l_post, C);
        if (k_tail == 1) gemv_n_kernel_fp32_fma<1>(lA, lB, N, k_tail, ldb, l_post, C);
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemv_fp32_fma(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
    const gemm_v_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typeV,
    const gemm_m_type_t typeH,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const gemm_post_t post,
    float *C)
{
    if (typeA == gemm_v_type::COL_VEC || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeH != gemm_m_type::EMPTY && typeH != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t num_threads = PPL_OMP_MAX_THREADS();

    if (num_threads == 1) {
        return gemv_operation_fp32_fma(
            A, B, V, H,
            typeA, typeB, typeV, typeH,
            N, K, ldb,
            alpha, beta, post, C);
    }

    const int64_t n_task_blk = N / num_threads;
    const int64_t n_task_tail = N % num_threads;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t t = 0; t < num_threads; ++t) {
        const int64_t nb = t * n_task_blk + (t < n_task_tail ? t : n_task_tail);
        const int64_t nb_eff = n_task_blk + (t < n_task_tail ? 1 : 0);

        const float *lA = A;
        const float *lB = B;
        if (typeB == gemm_m_type::NOTRANS) {
            lB += nb;
        } else {
            lB += nb * ldb;
        }

        const float *lV = V;
        if (typeV == gemm_v_type::ROW_VEC) {
            lV += nb;
        }

        const float *lH = H;
        if (typeH == gemm_m_type::NOTRANS) {
            lH += nb;
        }

        float *lC = C + nb;

        auto ret = gemv_operation_fp32_fma(
            lA, lB, lV, lH,
            typeA, typeB, typeV, typeH,
            nb_eff, K, ldb,
            alpha, beta, post, lC);

        (void) ret;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
