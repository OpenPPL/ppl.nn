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

#include "ppl/kernel/arm_server/gemm/neon/gemm.h"
#include "ppl/kernel/arm_server/common/internal_include.h"
#ifdef PPLNN_USE_AARCH64
#include "ppl/kernel/arm_server/gemm/neon/outer/gemm_ndarray_outer.h"
#endif

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#ifdef PPLNN_USE_AARCH64
template <typename eT>
static ppl::common::RetCode gemm_ndarray_common(
    const eT* A,
    const eT* B,
    const eT* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    eT* Y)
{
    if (A == nullptr || B == nullptr || Y == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((!transA && lda < K) || (transA && lda < M)) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((!transB && ldb < N) || (transB && ldb < K)) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ldy < N) {
        return ppl::common::RC_INVALID_VALUE;
    }

    // check C
    if (c_type > gemm_C_type::EMPTY && C == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (c_type >= gemm_C_type::VECTOR_W && ldc < N) {
        return ppl::common::RC_INVALID_VALUE;
    }

    return gemm_ndarray_common_outer<eT>(A, B, C, M, N, K, lda, ldb, ldc, transA, transB, alpha, beta, ldy, c_type, Y);
}
#endif

ppl::common::RetCode gemm_ndarray(
    const void* A,
    const void* B,
    const void* C,
    const ppl::common::datatype_t data_type,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    void* Y)
{
#ifdef PPLNN_USE_AARCH64
    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return gemm_ndarray_common<float>((const float*)A, (const float*)B, (const float*)C, M, N, K, lda, ldb, ldc, transA, transB, alpha, beta, ldy, c_type, (float*)Y);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return gemm_ndarray_common<__fp16>((const __fp16*)A, (const __fp16*)B, (const __fp16*)C, M, N, K, lda, ldb, ldc, transA, transB, alpha, beta, ldy, c_type, (__fp16*)Y);
#endif
        default: break;
    }
#endif

    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
