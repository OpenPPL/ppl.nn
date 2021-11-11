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

ppl::common::RetCode gemm_ref_fp32(
    const float *A,
    const float *B,
    const float *V, // vector C
    const float *H, // matrix C
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typeV,
    const gemm_m_type_t typeH,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldh,
    const float alpha,
    const float beta,
    const gemm_post_t post,
    float *Y)
{
    if (typeA == gemm_m_type::PACKED || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }
    const bool trans_A = typeA == gemm_m_type::TRANS;
    const bool trans_B = typeB == gemm_m_type::TRANS;
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
                    for (int64_t k = 0; k < K; ++k) {
                        y += A[m * lda + k] * B[k * ldb + n];
                    }
                }
                if (trans_A && !trans_B) { // KM, KN; TN
                    for (int64_t k = 0; k < K; ++k) {
                        y += A[k * lda + m] * B[k * ldb + n];
                    }
                }
                if (trans_A && trans_B) { // KM, NK; TT
                    for (int64_t k = 0; k < K; ++k) {
                        y += A[k * lda + m] * B[n * ldb + k];
                    }
                }
                if (!trans_A && trans_B) { // MK, NK; NT
                    for (int64_t k = 0; k < K; ++k) {
                        y += A[m * lda + k] * B[n * ldb + k];
                    }
                }
                y *= alpha;
            }
            if (beta != 0.0f) {
                if (V) {
                    if (typeV == gemm_v_type::ROW_VEC) y += beta * V[n];
                    if (typeV == gemm_v_type::COL_VEC) y += beta * V[m];
                    if (typeV == gemm_v_type::SCALAR) y += beta * V[0];
                }
                if (H) {
                    if (typeH == gemm_m_type::NOTRANS) y += beta * H[m * ldh + n];
                }
            }
            if (post & (gemm_post::RELU6 | gemm_post::RELU)) {
                y = max(y, 0.0f);
            }
            if (post & gemm_post::RELU6) {
                y = min(y, 6.0f);
            }
            Y[m * ldc + n] = y;
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
