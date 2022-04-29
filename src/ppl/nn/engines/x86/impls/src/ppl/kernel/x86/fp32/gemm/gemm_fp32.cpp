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

uint64_t gemm_fp32_get_packed_b_bytes(
    ppl::common::isa_t isa,
    const int64_t N,
    const int64_t K)
{
    if (isa & ppl::common::ISA_X86_AVX512) {
        return gemm_fp32_avx512_get_packed_b_bytes(N, K);
    }
    if (isa & ppl::common::ISA_X86_FMA) {
        return gemm_fp32_fma_get_packed_b_bytes(N, K);
    }
    return gemm_ref_fp32_get_packed_b_bytes(N, K);
}

ppl::common::RetCode gemm_pack_b_fp32(
    ppl::common::isa_t isa,
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    if (isa & ppl::common::ISA_X86_AVX512) {
        return gemm_pack_b_fp32_avx512(B, typeB, N, K, ldb, packedB);
    }
    if (isa & ppl::common::ISA_X86_FMA) {
        return gemm_pack_b_fp32_fma(B, typeB, N, K, ldb, packedB);
    }
    return gemm_ref_pack_b_fp32(B, typeB, N, K, ldb, packedB);
}

ppl::common::RetCode gemm_fp32(
    ppl::common::isa_t isa,
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
    if (isa & ppl::common::ISA_X86_AVX512) {
        return gemm_fp32_avx512(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
    }
    if (isa & ppl::common::ISA_X86_FMA) {
        return gemm_fp32_fma(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
    }
    return gemm_ref_fp32(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, C);
}

}}}; // namespace ppl::kernel::x86
