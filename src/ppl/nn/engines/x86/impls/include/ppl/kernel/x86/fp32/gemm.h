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

#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_H_

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/kernel/x86/common/gemm_common.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t gemm_fp32_get_packed_b_bytes(
    const ppl::common::isa_t isa,
    const int64_t N,
    const int64_t K);

ppl::common::RetCode gemm_fp32_pack_b(
    const ppl::common::isa_t isa,
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB);

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
    float *C);

ppl::common::RetCode batch_gemm_fp32(
    const ppl::common::isa_t isa,
    const float **A_list,
    const float **B_list,
    const float **bias_list,
    const float **sum_list,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t batch,
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
    float **C_list);

uint64_t gemm_fp32_ref_get_packed_b_bytes(
    const int64_t N,
    const int64_t K);

ppl::common::RetCode gemm_fp32_ref_pack_b(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB);

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
    float *C);

ppl::common::RetCode batch_gemm_fp32_ref(
    const float **A_list,
    const float **B_list,
    const float **bias_list,
    const float **sum_list,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t batch,
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
    float **C_list);

uint64_t gemm_fp32_sse_get_packed_b_bytes(
    const int64_t N,
    const int64_t K);

ppl::common::RetCode gemm_fp32_sse_pack_b(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB);

ppl::common::RetCode gemm_fp32_sse(
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
    float *C);

ppl::common::RetCode batch_gemm_fp32_sse(
    const float **A_list,
    const float **B_list,
    const float **bias_list,
    const float **sum_list,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t batch,
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
    float **C_list);

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
    float *C);

ppl::common::RetCode batch_gemv_fp32_sse(
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
    float **C_list);

uint64_t gemm_fp32_fma_get_packed_b_bytes(
    const int64_t N,
    const int64_t K);

ppl::common::RetCode gemm_fp32_fma_pack_b(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB);

ppl::common::RetCode gemm_fp32_fma(
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
    float *C);

ppl::common::RetCode batch_gemm_fp32_fma(
    const float **A_list,
    const float **B_list,
    const float **bias_list,
    const float **sum_list,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t batch,
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
    float **C_list);

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
    float *C);

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
    float **C_list);

#ifdef PPL_USE_X86_AVX512
uint64_t gemm_fp32_avx512_get_packed_b_bytes(
    const int64_t N,
    const int64_t K);

ppl::common::RetCode gemm_fp32_avx512_pack_b(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB);

ppl::common::RetCode gemm_fp32_avx512(
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
    float *C);

ppl::common::RetCode batch_gemm_fp32_avx512(
    const float **A_list,
    const float **B_list,
    const float **bias_list,
    const float **sum_list,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t batch,
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
    float **C_list);
#endif

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_GEMM_H_
