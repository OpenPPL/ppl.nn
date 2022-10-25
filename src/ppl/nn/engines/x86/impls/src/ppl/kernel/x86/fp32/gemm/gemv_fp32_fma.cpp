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
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

template<int64_t u_n>
void gemv_t_kernel_fp32_fma(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float *C)
{
    const float *lB = B;
    const float *lbias = bias;
    const float *lsum = sum;
    float *lC = C;

    const int64_t K_REG_ELTS = 8;
    const int64_t u_k = 2 * K_REG_ELTS;
    const bool apply_beta = beta != 0.0f;
    for (int64_t n = 0; n < N; n += u_n) {
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
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
                ymm4 = _mm256_loadu_ps(rA + 0 * K_REG_ELTS);
                ymm5 = _mm256_loadu_ps(rA + 1 * K_REG_ELTS);
                if (u_n > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 0 * K_REG_ELTS), ymm4, ymm0);
                if (u_n > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 0 * K_REG_ELTS), ymm4, ymm1);
                if (u_n > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 0 * K_REG_ELTS), ymm4, ymm2);
                if (u_n > 3) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 0 * K_REG_ELTS), ymm4, ymm3);
                if (u_n > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 0 * ldb + 1 * K_REG_ELTS), ymm5, ymm0);
                if (u_n > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(rB0 + 1 * ldb + 1 * K_REG_ELTS), ymm5, ymm1);
                if (u_n > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 0 * ldb + 1 * K_REG_ELTS), ymm5, ymm2);
                if (u_n > 3) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(rB2 + 1 * ldb + 1 * K_REG_ELTS), ymm5, ymm3);
                rA += u_k;
                if (u_n > 0) rB0 += u_k;
                if (u_n > 2) rB2 += u_k;
            }
#define _MM256_RSUM_PS(YMM, SREG) do {\
    float v[K_REG_ELTS];\
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
            if (u_n > 0) r0 += beta * lC[0];
            if (u_n > 1) r1 += beta * lC[1];
            if (u_n > 2) r2 += beta * lC[2];
            if (u_n > 3) r3 += beta * lC[3];
        }
        if (typebias == gemm_v_type::COL_VEC || typebias == gemm_v_type::SCALAR) {
            if (u_n > 0) r0 += beta_bias * lbias[0];
            if (u_n > 1) r1 += beta_bias * lbias[0];
            if (u_n > 2) r2 += beta_bias * lbias[0];
            if (u_n > 3) r3 += beta_bias * lbias[0];
        }
        if (typebias == gemm_v_type::ROW_VEC) {
            if (u_n > 0) r0 += beta_bias * lbias[0];
            if (u_n > 1) r1 += beta_bias * lbias[1];
            if (u_n > 2) r2 += beta_bias * lbias[2];
            if (u_n > 3) r3 += beta_bias * lbias[3];
            lbias += u_n;
        }
        if (typesum == gemm_m_type::NOTRANS) {
            if (u_n > 0) r0 += beta_sum * lsum[0];
            if (u_n > 1) r1 += beta_sum * lsum[1];
            if (u_n > 2) r2 += beta_sum * lsum[2];
            if (u_n > 3) r3 += beta_sum * lsum[3];
            lsum += u_n;
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
    const float alpha,
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

        float a0, a1, a2, a3;
        if (u_k > 0) a0 = lA[0] * alpha;
        if (u_k > 1) a1 = lA[1] * alpha;
        if (u_k > 2) a2 = lA[2] * alpha;
        if (u_k > 3) a3 = lA[3] * alpha;
        if (u_k > 0) ymm0 = _mm256_set1_ps(a0);
        if (u_k > 1) ymm1 = _mm256_set1_ps(a1);
        if (u_k > 2) ymm2 = _mm256_set1_ps(a2);
        if (u_k > 3) ymm3 = _mm256_set1_ps(a3);

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
            if (u_k > 0) r = rB0[0 * ldb] * a0;
            if (u_k > 1) r += rB0[1 * ldb] * a1;
            if (u_k > 2) r += rB2[0 * ldb] * a2;
            if (u_k > 3) r += rB2[1 * ldb] * a3;
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

template<gemm_v_type_t typesum, gemm_m_type_t typebias>
void gemv_fp32_apply_beta_fma(
    const float *bias,
    const float *sum,
    const int64_t N,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    float *C)
{
    if (typesum == gemm_v_type::EMPTY && typebias == gemm_m_type::EMPTY && beta == 0.0f) {
        memset32_avx(C, 0, N);
        return;
    }

    const int64_t N_REG_ELTS = 8;
    const int64_t unroll_n = N_REG_ELTS * 2;
 
    __m256 ymm_v, ymm_beta, ymm_beta_bias, ymm_beta_sum;;
    ymm_beta = _mm256_set1_ps(beta);
    ymm_beta_bias = _mm256_set1_ps(beta_bias);
    ymm_beta_sum = _mm256_set1_ps(beta_sum);

    
    if (typebias == gemm_v_type::SCALAR || typebias == gemm_v_type::COL_VEC) {
        ymm_v = _mm256_mul_ps(_mm256_set1_ps(bias[0]), ymm_beta_bias);
    }

    bool apply_beta = beta != 0.0f;

    const float *lbias = bias;
    const float *lsum = sum;
    float *lC = C;
    int64_t n = N;
    while (n >= unroll_n) {
        __m256 ymm0, ymm1;
        ymm0 = _mm256_setzero_ps();
        ymm1 = _mm256_setzero_ps();
        if (typebias != gemm_v_type::EMPTY) {
            if (typebias == gemm_v_type::ROW_VEC) {
                ymm0 = _mm256_mul_ps(_mm256_loadu_ps(lbias + 0 * N_REG_ELTS), ymm_beta_bias);
                ymm1 = _mm256_mul_ps(_mm256_loadu_ps(lbias + 1 * N_REG_ELTS), ymm_beta_bias);
                lbias += unroll_n;
            } else {
                ymm0 = ymm_v;
                ymm1 = ymm_v;
            }
        }
        if (typesum == gemm_m_type::NOTRANS) {
            ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(lsum + 0 * N_REG_ELTS), ymm_beta_sum, ymm0);
            ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(lsum + 1 * N_REG_ELTS), ymm_beta_sum, ymm1);
            lsum += unroll_n;
        }
        if (apply_beta) {
            ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(lC + 0 * N_REG_ELTS), ymm_beta, ymm0);
            ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(lC + 1 * N_REG_ELTS), ymm_beta, ymm1);
        }
        _mm256_storeu_ps(lC + 0 * N_REG_ELTS, ymm0);
        _mm256_storeu_ps(lC + 1 * N_REG_ELTS, ymm1);
        lC += unroll_n;
        n -= unroll_n;
    }
    while (n > 0) {
        float y = 0.0f;
        if (typebias != gemm_v_type::EMPTY) {
            if (typebias == gemm_v_type::ROW_VEC) {
                y = beta_bias * lbias[0];
                lbias += 1;
            } else {
                y = beta_bias * bias[0];
            }
        }
        if (typesum == gemm_m_type::NOTRANS) {
            y += beta_sum * lsum[0];
            lsum += 1;
        }
        if (apply_beta) {
            y += beta * lC[0];
        }
        lC[0] = y;
        lC += 1;
        n -= 1;
    }
}

ppl::common::RetCode gemv_operation_fp32_fma(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_v_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float *C)
{
    if (typeA == gemm_v_type::COL_VEC || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool apply_alpha = alpha != 0.0f && typeA != gemm_v_type::EMPTY && typeB != gemm_m_type::EMPTY;
    const bool apply_betas = beta != 0.0f || (beta_bias != 0.0f && typebias != gemm_v_type::EMPTY) || (beta_sum != 0.0f && typesum != gemm_m_type::EMPTY);

    if (!apply_alpha && !apply_betas) {
        memset32_avx(C, 0, N);
        return ppl::common::RC_SUCCESS;
    }

    if (typeB == gemm_m_type::TRANS || K == 0) {
        const int64_t MAX_U_N = 4;
        const int64_t n_body = round(N, MAX_U_N);
        const int64_t n_tail = N - n_body;

        const float *lB = B;
        const float *lbias = bias;
        const float *lsum = sum;
        float *lC = C;
        if (n_body) gemv_t_kernel_fp32_fma<MAX_U_N>(A, lB, lbias, lsum, typebias, typesum, n_body, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
        
        lB += n_body * ldb;
        lC += n_body;

        if (typebias == gemm_v_type::ROW_VEC) {
            lbias += n_body;
        }
        if (typesum == gemm_m_type::NOTRANS) {
            lsum += n_body;
        }
        
        if (n_tail == 3) gemv_t_kernel_fp32_fma<3>(A, lB, lbias, lsum, typebias, typesum, n_tail, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
        if (n_tail == 2) gemv_t_kernel_fp32_fma<2>(A, lB, lbias, lsum, typebias, typesum, n_tail, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
        if (n_tail == 1) gemv_t_kernel_fp32_fma<1>(A, lB, lbias, lsum, typebias, typesum, n_tail, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
    } else {
        auto apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
        if (typesum == gemm_m_type::NOTRANS) {
            if (typebias == gemm_v_type::EMPTY) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::NOTRANS, gemm_v_type::EMPTY>;
            if (typebias == gemm_v_type::SCALAR) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::NOTRANS, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::NOTRANS, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::NOTRANS, gemm_v_type::ROW_VEC>;
        } else {
            if (typebias == gemm_v_type::EMPTY) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
            if (typebias == gemm_v_type::SCALAR) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::EMPTY, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::EMPTY, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_beta_func = gemv_fp32_apply_beta_fma<gemm_m_type::EMPTY, gemm_v_type::ROW_VEC>;
        }
        apply_beta_func(bias, sum, N, beta, beta_bias, beta_sum, C);

        const int64_t MAX_U_K = 4;
        const int64_t k_body = round(K, MAX_U_K);
        const int64_t k_tail = K - k_body;

        const float *lA = A;
        const float *lB = B;

        gemm_post_t l_post = gemm_post::NONE;
        if (!k_tail) l_post = post;

        if (k_body) gemv_n_kernel_fp32_fma<MAX_U_K>(lA, lB, N, k_body, ldb, alpha, l_post, C);

        lA += k_body;
        lB += k_body * ldb;

        l_post = post;

        if (k_tail == 3) gemv_n_kernel_fp32_fma<3>(lA, lB, N, k_tail, ldb, alpha, l_post, C);
        if (k_tail == 2) gemv_n_kernel_fp32_fma<2>(lA, lB, N, k_tail, ldb, alpha, l_post, C);
        if (k_tail == 1) gemv_n_kernel_fp32_fma<1>(lA, lB, N, k_tail, ldb, alpha, l_post, C);
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemv_fp32_fma(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_v_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float *C)
{
    if (typeA == gemm_v_type::COL_VEC || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t num_threads = PPL_OMP_MAX_THREADS();

    if (num_threads == 1) {
        return gemv_operation_fp32_fma(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            N, K, ldb,
            alpha, beta, beta_bias, beta_sum, post, C);
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

        const float *lbias = bias;
        if (typebias == gemm_v_type::ROW_VEC) {
            lbias += nb;
        }

        const float *lsum = sum;
        if (typesum == gemm_m_type::NOTRANS) {
            lsum += nb;
        }

        float *lC = C + nb;

        auto ret = gemv_operation_fp32_fma(
            lA, lB, lbias, lsum,
            typeA, typeB, typebias, typesum,
            nb_eff, K, ldb,
            alpha, beta, beta_bias, beta_sum, post, lC);

        (void) ret;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode batch_gemv_fp32_fma(
    const float **A_list,
    const float **B_list,
    const float **bias_list,
    const float **sum_list,
    const gemm_v_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t batch,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float **C_list)
{
    if (batch == 1) {
        return gemv_fp32_fma(
            *A_list, *B_list, bias_list ? *bias_list : nullptr, sum_list ? *sum_list : nullptr,
            typeA, typeB, typebias, typesum,
            N, K, ldb,
            alpha, beta, beta_bias, beta_sum, post, *C_list);
    }

    if (typeA == gemm_v_type::COL_VEC || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t num_threads = PPL_OMP_MAX_THREADS();

    if (num_threads == 1) {
        for (int64_t b = 0; b < batch; ++b) {
            auto ret = gemv_operation_fp32_fma(
                A_list[b], B_list[b], bias_list ? bias_list[b] : nullptr, sum_list ? sum_list[b] : nullptr,
                typeA, typeB, typebias, typesum,
                N, K, ldb,
                alpha, beta, beta_bias, beta_sum, post, C_list[b]);
            if (ppl::common::RC_SUCCESS != ret) {
                return ret;
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    const int64_t n_threads = div_up(num_threads, batch);

    const int64_t n_task_blk = N / n_threads;
    const int64_t n_task_tail = N % n_threads;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t t = 0; t < batch * n_threads; ++t) {
        const int64_t b = t / n_threads;
        const int64_t nt = t % n_threads;
        const int64_t nb = nt * n_task_blk + (nt < n_task_tail ? nt : n_task_tail);
        const int64_t nb_eff = n_task_blk + (nt < n_task_tail ? 1 : 0);

        const float *lA = A_list[b];
        const float *lB = B_list[b];
        if (typeB == gemm_m_type::NOTRANS) {
            lB += nb;
        } else {
            lB += nb * ldb;
        }

        const float *lbias = bias_list ? bias_list[b] : nullptr;
        if (typebias == gemm_v_type::ROW_VEC) {
            lbias += nb;
        }

        const float *lsum = sum_list ? sum_list[b] : nullptr;
        if (typesum == gemm_m_type::NOTRANS) {
            lsum += nb;
        }

        float *lC = C_list[b] + nb;

        auto ret = gemv_operation_fp32_fma(
            lA, lB, lbias, lsum,
            typeA, typeB, typebias, typesum,
            nb_eff, K, ldb,
            alpha, beta, beta_bias, beta_sum, post, lC);

        (void) ret;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
