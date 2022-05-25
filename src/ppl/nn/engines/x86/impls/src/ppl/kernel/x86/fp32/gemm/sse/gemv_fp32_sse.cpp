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
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

template<int64_t u_n>
void gemv_t_kernel_fp32_sse(
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

    const int64_t K_REG_ELTS = 4;
    const int64_t u_k = 2 * K_REG_ELTS;
    const bool apply_beta = beta != 0.0f;
    for (int64_t n = 0; n < N; n += u_n) {
        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;
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
            if (u_n > 0) xmm0 = _mm_setzero_ps();
            if (u_n > 1) xmm1 = _mm_setzero_ps();
            if (u_n > 2) xmm2 = _mm_setzero_ps();
            if (u_n > 3) xmm3 = _mm_setzero_ps();
            for (; k <= K - u_k; k += u_k) {
                xmm4 = _mm_loadu_ps(rA + 0 * K_REG_ELTS);
                xmm5 = _mm_loadu_ps(rA + 1 * K_REG_ELTS);
                if (u_n > 0) xmm0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 0 * K_REG_ELTS), xmm4), xmm0);
                if (u_n > 1) xmm1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 0 * K_REG_ELTS), xmm4), xmm1);
                if (u_n > 2) xmm2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 0 * K_REG_ELTS), xmm4), xmm2);
                if (u_n > 3) xmm3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 0 * K_REG_ELTS), xmm4), xmm3);
                if (u_n > 0) xmm0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 1 * K_REG_ELTS), xmm5), xmm0);
                if (u_n > 1) xmm1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 1 * K_REG_ELTS), xmm5), xmm1);
                if (u_n > 2) xmm2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 1 * K_REG_ELTS), xmm5), xmm2);
                if (u_n > 3) xmm3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 1 * K_REG_ELTS), xmm5), xmm3);
                rA += u_k;
                if (u_n > 0) rB0 += u_k;
                if (u_n > 2) rB2 += u_k;
            }
#define _mm_RSUM_PS(xmm, SREG) do {\
    float v[K_REG_ELTS];\
    _mm_storeu_ps(v, (xmm));\
    (SREG) = ((v[0] + v[1]) + (v[2] + v[3]));\
} while(0)
            if (u_n > 0) _mm_RSUM_PS(xmm0, r0);
            if (u_n > 1) _mm_RSUM_PS(xmm1, r1);
            if (u_n > 2) _mm_RSUM_PS(xmm2, r2);
            if (u_n > 3) _mm_RSUM_PS(xmm3, r3);
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
void gemv_n_kernel_fp32_sse(
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
    __m128 xmm_v0 = _mm_set1_ps(0.0f);
    __m128 xmm_v6 = _mm_set1_ps(6.0f);

    const int64_t N_REG_ELTS = 4;
    const int64_t u_n = N_REG_ELTS * 4;
    for (int64_t k = 0; k < K; k += u_k) {
        __m128 xmm0, xmm1, xmm2, xmm3;
        __m128 xmm4, xmm5, xmm6, xmm7;

        const bool do_relu_max = k + u_k >= K && (post == gemm_post::RELU || post == gemm_post::RELU6);
        const bool do_relu_min = k + u_k >= K && post == gemm_post::RELU6;

        float a0, a1, a2, a3;
        if (u_k > 0) a0 = lA[0] * alpha;
        if (u_k > 1) a1 = lA[1] * alpha;
        if (u_k > 2) a2 = lA[2] * alpha;
        if (u_k > 3) a3 = lA[3] * alpha;
        if (u_k > 0) xmm0 = _mm_set1_ps(a0);
        if (u_k > 1) xmm1 = _mm_set1_ps(a1);
        if (u_k > 2) xmm2 = _mm_set1_ps(a2);
        if (u_k > 3) xmm3 = _mm_set1_ps(a3);

        float *rC = C;
        const float *rB0;
        const float *rB2;
        if (u_k > 0) rB0 = lB + 0 * ldb;
        if (u_k > 2) rB2 = lB + 2 * ldb;
        int64_t n = N;
        while (n >= u_n) {
            if (u_k > 0) {
                xmm4 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 0 * N_REG_ELTS), xmm0);
                xmm5 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 1 * N_REG_ELTS), xmm0);
                xmm6 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 2 * N_REG_ELTS), xmm0);
                xmm7 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 3 * N_REG_ELTS), xmm0);
            }
            if (u_k > 1) {
                xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 0 * N_REG_ELTS), xmm1), xmm4);
                xmm5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 1 * N_REG_ELTS), xmm1), xmm5);
                xmm6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 2 * N_REG_ELTS), xmm1), xmm6);
                xmm7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 3 * N_REG_ELTS), xmm1), xmm7);
            }
            if (u_k > 2) {
                xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 0 * N_REG_ELTS), xmm2), xmm4);
                xmm5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 1 * N_REG_ELTS), xmm2), xmm5);
                xmm6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 2 * N_REG_ELTS), xmm2), xmm6);
                xmm7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 3 * N_REG_ELTS), xmm2), xmm7);
            }
            if (u_k > 3) {
                xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 0 * N_REG_ELTS), xmm3), xmm4);
                xmm5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 1 * N_REG_ELTS), xmm3), xmm5);
                xmm6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 2 * N_REG_ELTS), xmm3), xmm6);
                xmm7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 3 * N_REG_ELTS), xmm3), xmm7);
            }
            xmm4 = _mm_add_ps(_mm_loadu_ps(rC + 0 * N_REG_ELTS), xmm4);
            xmm5 = _mm_add_ps(_mm_loadu_ps(rC + 1 * N_REG_ELTS), xmm5);
            xmm6 = _mm_add_ps(_mm_loadu_ps(rC + 2 * N_REG_ELTS), xmm6);
            xmm7 = _mm_add_ps(_mm_loadu_ps(rC + 3 * N_REG_ELTS), xmm7);
            if (do_relu_max) {
                xmm4 = _mm_max_ps(xmm4, xmm_v0);
                xmm5 = _mm_max_ps(xmm5, xmm_v0);
                xmm6 = _mm_max_ps(xmm6, xmm_v0);
                xmm7 = _mm_max_ps(xmm7, xmm_v0);
            }
            if (do_relu_min) {
                xmm4 = _mm_min_ps(xmm4, xmm_v6);
                xmm5 = _mm_min_ps(xmm5, xmm_v6);
                xmm6 = _mm_min_ps(xmm6, xmm_v6);
                xmm7 = _mm_min_ps(xmm7, xmm_v6);
            }
            _mm_storeu_ps(rC + 0 * N_REG_ELTS, xmm4);
            _mm_storeu_ps(rC + 1 * N_REG_ELTS, xmm5);
            _mm_storeu_ps(rC + 2 * N_REG_ELTS, xmm6);
            _mm_storeu_ps(rC + 3 * N_REG_ELTS, xmm7);
            rC += u_n;
            if (u_k > 0) rB0 += u_n;
            if (u_k > 2) rB2 += u_n;
            n -= u_n;
        }
        if (n >= N_REG_ELTS * 2) {
            if (u_k > 0) {
                xmm4 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 0 * N_REG_ELTS), xmm0);
                xmm5 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 1 * N_REG_ELTS), xmm0);
            }
            if (u_k > 1) {
                xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 0 * N_REG_ELTS), xmm1), xmm4);
                xmm5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 1 * N_REG_ELTS), xmm1), xmm5);
            }
            if (u_k > 2) {
                xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 0 * N_REG_ELTS), xmm2), xmm4);
                xmm5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 1 * N_REG_ELTS), xmm2), xmm5);
            }
            if (u_k > 3) {
                xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 0 * N_REG_ELTS), xmm3), xmm4);
                xmm5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 1 * N_REG_ELTS), xmm3), xmm5);
            }
            xmm4 = _mm_add_ps(_mm_loadu_ps(rC + 0 * N_REG_ELTS), xmm4);
            xmm5 = _mm_add_ps(_mm_loadu_ps(rC + 1 * N_REG_ELTS), xmm5);
            if (do_relu_max) {
                xmm4 = _mm_max_ps(xmm4, xmm_v0);
                xmm5 = _mm_max_ps(xmm5, xmm_v0);
            }
            if (do_relu_min) {
                xmm4 = _mm_min_ps(xmm4, xmm_v6);
                xmm5 = _mm_min_ps(xmm5, xmm_v6);
            }
            _mm_storeu_ps(rC + 0 * N_REG_ELTS, xmm4);
            _mm_storeu_ps(rC + 1 * N_REG_ELTS, xmm5);
            rC += N_REG_ELTS * 2;
            if (u_k > 0) rB0 += N_REG_ELTS * 2;
            if (u_k > 2) rB2 += N_REG_ELTS * 2;
            n -= N_REG_ELTS * 2;
        }
        if (n >= N_REG_ELTS) {
            if (u_k > 0) xmm4 = _mm_mul_ps(_mm_loadu_ps(rB0 + 0 * ldb + 0 * N_REG_ELTS), xmm0);
            if (u_k > 1) xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB0 + 1 * ldb + 0 * N_REG_ELTS), xmm1), xmm4);
            if (u_k > 2) xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 0 * ldb + 0 * N_REG_ELTS), xmm2), xmm4);
            if (u_k > 3) xmm4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(rB2 + 1 * ldb + 0 * N_REG_ELTS), xmm3), xmm4);
            xmm4 = _mm_add_ps(_mm_loadu_ps(rC + 0 * N_REG_ELTS), xmm4);
            if (do_relu_max) xmm4 = _mm_max_ps(xmm4, xmm_v0);
            if (do_relu_min) xmm4 = _mm_min_ps(xmm4, xmm_v6);
            _mm_storeu_ps(rC + 0 * N_REG_ELTS, xmm4);
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
void gemv_fp32_apply_beta_sse(
    const float *bias,
    const float *sum,
    const int64_t N,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    float *C)
{
    if (typesum == gemm_v_type::EMPTY && typebias == gemm_m_type::EMPTY && beta == 0.0f) {
        memset32_sse(C, 0, N);
        return;
    }

    const int64_t N_REG_ELTS = 4;
    const int64_t unroll_n = N_REG_ELTS * 2;
 
    __m128 xmm_v, xmm_beta, xmm_beta_bias, xmm_beta_sum;;
    xmm_beta = _mm_set1_ps(beta);
    xmm_beta_bias = _mm_set1_ps(beta_bias);
    xmm_beta_sum = _mm_set1_ps(beta_sum);

    
    if (typebias == gemm_v_type::SCALAR || typebias == gemm_v_type::COL_VEC) {
        xmm_v = _mm_mul_ps(_mm_set1_ps(bias[0]), xmm_beta_bias);
    }

    bool apply_beta = beta != 0.0f;

    const float *lbias = bias;
    const float *lsum = sum;
    float *lC = C;
    int64_t n = N;
    while (n >= unroll_n) {
        __m128 xmm0, xmm1;
        xmm0 = _mm_setzero_ps();
        xmm1 = _mm_setzero_ps();
        if (typebias != gemm_v_type::EMPTY) {
            if (typebias == gemm_v_type::ROW_VEC) {
                xmm0 = _mm_mul_ps(_mm_loadu_ps(lbias + 0 * N_REG_ELTS), xmm_beta_bias);
                xmm1 = _mm_mul_ps(_mm_loadu_ps(lbias + 1 * N_REG_ELTS), xmm_beta_bias);
                lbias += unroll_n;
            } else {
                xmm0 = xmm_v;
                xmm1 = xmm_v;
            }
        }
        if (typesum == gemm_m_type::NOTRANS) {
            xmm0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(lsum + 0 * N_REG_ELTS), xmm_beta_sum), xmm0);
            xmm1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(lsum + 1 * N_REG_ELTS), xmm_beta_sum), xmm1);
            lsum += unroll_n;
        }
        if (apply_beta) {
            xmm0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(lC + 0 * N_REG_ELTS), xmm_beta), xmm0);
            xmm1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(lC + 1 * N_REG_ELTS), xmm_beta), xmm1);
        }
        _mm_storeu_ps(lC + 0 * N_REG_ELTS, xmm0);
        _mm_storeu_ps(lC + 1 * N_REG_ELTS, xmm1);
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

ppl::common::RetCode gemv_operation_fp32_sse(
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
        memset32_sse(C, 0, N);
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
        if (n_body) gemv_t_kernel_fp32_sse<MAX_U_N>(A, lB, lbias, lsum, typebias, typesum, n_body, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
        
        lB += n_body * ldb;
        lC += n_body;

        if (typebias == gemm_v_type::ROW_VEC) {
            lbias += n_body;
        }
        if (typesum == gemm_m_type::NOTRANS) {
            lsum += n_body;
        }
        
        if (n_tail == 3) gemv_t_kernel_fp32_sse<3>(A, lB, lbias, lsum, typebias, typesum, n_tail, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
        if (n_tail == 2) gemv_t_kernel_fp32_sse<2>(A, lB, lbias, lsum, typebias, typesum, n_tail, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
        if (n_tail == 1) gemv_t_kernel_fp32_sse<1>(A, lB, lbias, lsum, typebias, typesum, n_tail, K, ldb, alpha, beta, beta_bias, beta_sum, post, lC);
    } else {
        auto apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
        if (typesum == gemm_m_type::NOTRANS) {
            if (typebias == gemm_v_type::EMPTY) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::NOTRANS, gemm_v_type::EMPTY>;
            if (typebias == gemm_v_type::SCALAR) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::NOTRANS, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::NOTRANS, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::NOTRANS, gemm_v_type::ROW_VEC>;
        } else {
            if (typebias == gemm_v_type::EMPTY) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
            if (typebias == gemm_v_type::SCALAR) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::EMPTY, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::EMPTY, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_beta_func = gemv_fp32_apply_beta_sse<gemm_m_type::EMPTY, gemm_v_type::ROW_VEC>;
        }
        apply_beta_func(bias, sum, N, beta, beta_bias, beta_sum, C);

        const int64_t MAX_U_K = 4;
        const int64_t k_body = round(K, MAX_U_K);
        const int64_t k_tail = K - k_body;

        const float *lA = A;
        const float *lB = B;

        gemm_post_t l_post = gemm_post::NONE;
        if (!k_tail) l_post = post;

        if (k_body) gemv_n_kernel_fp32_sse<MAX_U_K>(lA, lB, N, k_body, ldb, alpha, l_post, C);

        lA += k_body;
        lB += k_body * ldb;

        l_post = post;

        if (k_tail == 3) gemv_n_kernel_fp32_sse<3>(lA, lB, N, k_tail, ldb, alpha, l_post, C);
        if (k_tail == 2) gemv_n_kernel_fp32_sse<2>(lA, lB, N, k_tail, ldb, alpha, l_post, C);
        if (k_tail == 1) gemv_n_kernel_fp32_sse<1>(lA, lB, N, k_tail, ldb, alpha, l_post, C);
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemv_fp32_sse(
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
        return gemv_operation_fp32_sse(
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

        auto ret = gemv_operation_fp32_sse(
            lA, lB, lbias, lsum,
            typeA, typeB, typebias, typesum,
            nb_eff, K, ldb,
            alpha, beta, beta_bias, beta_sum, post, lC);

        (void) ret;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
