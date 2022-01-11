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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N4CX_SGEMM_N4CX_SGEMM_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CONV2D_NEON_FP16_N4CX_SGEMM_N4CX_SGEMM_H_

#include <cstdint>

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

enum class N4cxSgemmBlockingOrd {
    M_N_K,
    N_M_K,
    N_K_M
};

void sgemm_n4cx_inner_blocking_4x4_fp32(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k);

void sgemm_n4cx_inner_blocking_8x4_fp32(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k);

template <N4cxSgemmBlockingOrd order>
void sgemm_n4cx_blocking_fp32(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

template <>
void sgemm_n4cx_blocking_fp32<N4cxSgemmBlockingOrd::M_N_K>(
    const float *a,
    float *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1);

typedef void (*sgemm_n4cx_kernel_func_t)(
    const float *A,
    const float *B,
    const float *constant,
    const float *DX,
    float *C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t lddx,
    const int64_t ldc);

extern const sgemm_n4cx_kernel_func_t sgemm_n4cx_kernel_m4nx_fp32_func_table[12][3][6];
extern const sgemm_n4cx_kernel_func_t sgemm_n4cx_kernel_m8nx_fp32_func_table[10][3][6];

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
