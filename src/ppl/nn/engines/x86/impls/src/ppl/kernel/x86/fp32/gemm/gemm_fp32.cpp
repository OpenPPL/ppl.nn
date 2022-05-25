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
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t gemm_fp32_ref_get_packed_b_bytes(
    const int64_t N,
    const int64_t K)
{
    return sizeof(float) * K * N;
}

ppl::common::RetCode gemm_fp32_ref_pack_b(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    if (N <= 0 || K <= 0) {
        return ppl::common::RC_SUCCESS;
    }

    const bool is_trans_b = typeB == gemm_m_type::TRANS;

    if (is_trans_b) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                packedB[n * K + k] = B[n * ldb + k];
            }
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                packedB[n * K + k] = B[k * ldb + n];
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemm_fp32_ref(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float *C)
{
    if (typeA == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }
    const bool trans_A = typeA == gemm_m_type::TRANS;
    const bool trans_B = typeB == gemm_m_type::TRANS || typeB == gemm_m_type::PACKED;
    const int64_t lldb = typeB == gemm_m_type::PACKED ? K : ldb;
#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t m = 0; m < M; ++m) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t n = 0; n < N; ++n) {
            float y = 0.0f;
            if (alpha != 0.0f && typeA != gemm_m_type::EMPTY && typeB != gemm_m_type::EMPTY) {
                if (!trans_A && !trans_B) { // MK, KN; NN
                    for (int64_t k = 0; k < K; ++k)
                        y += A[m * lda + k] * B[k * lldb + n];
                }
                if (trans_A && !trans_B) { // KM, KN; TN
                    for (int64_t k = 0; k < K; ++k)
                        y += A[k * lda + m] * B[k * lldb + n];
                }
                if (trans_A && trans_B) { // KM, NK; TT
                    for (int64_t k = 0; k < K; ++k)
                        y += A[k * lda + m] * B[n * lldb + k];
                }
                if (!trans_A && trans_B) { // MK, NK; NT
                    for (int64_t k = 0; k < K; ++k)
                        y += A[m * lda + k] * B[n * lldb + k];
                }
                y *= alpha;
            }
            if (beta != 0.0f) y += beta * C[m * ldc + n];
            if (typebias == gemm_v_type::ROW_VEC) y += beta_bias * bias[n];
            if (typebias == gemm_v_type::COL_VEC) y += beta_bias * bias[m];
            if (typebias == gemm_v_type::SCALAR) y += beta_bias * bias[0];
            if (typesum == gemm_m_type::NOTRANS) y += beta_sum * sum[m * ldsum + n];
            if (post & (gemm_post::RELU6 | gemm_post::RELU)) y = max(y, 0.0f);
            if (post & gemm_post::RELU6) y = min(y, 6.0f);
            C[m * ldc + n] = y;
        }
    }
    return ppl::common::RC_SUCCESS;
}

uint64_t gemm_fp32_get_packed_b_bytes(
    const ppl::common::isa_t isa,
    const int64_t N,
    const int64_t K)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return gemm_fp32_avx512_get_packed_b_bytes(N, K);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return gemm_fp32_fma_get_packed_b_bytes(N, K);
    }
    if (isa & ppl::common::ISA_X86_SSE) {
        return gemm_fp32_sse_get_packed_b_bytes(N, K);
    }
    return gemm_fp32_ref_get_packed_b_bytes(N, K);
}

ppl::common::RetCode gemm_fp32_pack_b(
    const ppl::common::isa_t isa,
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return gemm_fp32_avx512_pack_b(B, typeB, N, K, ldb, packedB);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return gemm_fp32_fma_pack_b(B, typeB, N, K, ldb, packedB);
    }
    if (isa & ppl::common::ISA_X86_SSE) {
        return gemm_fp32_sse_pack_b(B, typeB, N, K, ldb, packedB);
    }
    return gemm_fp32_ref_pack_b(B, typeB, N, K, ldb, packedB);
}

ppl::common::RetCode gemm_fp32(
    const ppl::common::isa_t isa,
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float *C)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return gemm_fp32_avx512(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return gemm_fp32_fma(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
    }
    if (isa & ppl::common::ISA_X86_SSE) {
        return gemm_fp32_sse(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
    }
    return gemm_fp32_ref(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
}

}}}; // namespace ppl::kernel::x86
