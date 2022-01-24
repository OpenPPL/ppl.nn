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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N8CX_HGEMM_N8CX_HGEMM_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N8CX_HGEMM_N8CX_HGEMM_H_

#ifdef PPLNN_USE_ARMV8_2_FP16

#include <cstdint>

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

enum class N8cxHgemmBlockingOrd {
    M_N_K,
    N_M_K,
    N_K_M
};

void hgemm_n8cx_inner_blocking_8x8_fp16(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k);

void hgemm_n8cx_inner_blocking_16x8_fp16(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k);

template <N8cxHgemmBlockingOrd order>
void hgemm_n8cx_blocking_fp16(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

template <>
void hgemm_n8cx_blocking_fp16<N8cxHgemmBlockingOrd::M_N_K>(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

typedef void (*hgemm_n8cx_kernel_fp16_func_t)(
    const __fp16 *A,
    const __fp16 *B,
    const __fp16 *constant,
    const __fp16 *DX,
    __fp16 *C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t lddx,
    const int64_t ldc);

extern const hgemm_n8cx_kernel_fp16_func_t hgemm_n8cx_kernel_m8nx_fp16_func_table[12][3][6];
extern const hgemm_n8cx_kernel_fp16_func_t hgemm_n8cx_kernel_m16nx_fp16_func_table[10][3][6];

}}}}; // namespace ppl::kernel::arm_server::neon

#endif

#endif
