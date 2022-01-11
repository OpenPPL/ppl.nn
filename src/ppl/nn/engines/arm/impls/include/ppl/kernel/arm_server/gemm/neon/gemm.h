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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_GEMM_H_
#define __ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_GEMM_H_

#include "ppl/kernel/arm_server/common/general_include.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

class gemm_fuse_flag {
public:
    enum {
        NONE = 0,
        RELU = 1 << 0,
    };
};
typedef uint32_t gemm_fuse_flag_t;

class gemm_C_type {
public:
    enum {
        EMPTY    = 0,
        // SCALAR   = 1,
        // VECTOR_H = 2,
        // VECTOR_W = 3,
        MATRIX   = 4,
    };
};
typedef uint32_t gemm_C_type_t;

#ifdef PPL_USE_ARM_SERVER_FP16
size_t ppl_arm_server_kernel_fp16_gemm_get_buffer_size(
    const int64_t sgemm_m1,
    const int64_t sgemm_n1);

ppl::common::RetCode gemm_fp16(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *c,
    __fp16 *y,
    __fp16 *tmp_buffer,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldy,
    const __fp16 alpha,
    const __fp16 beta,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3);
#endif

size_t ppl_arm_server_kernel_fp32_gemm_get_buffer_size(
    const int64_t sgemm_m1,
    const int64_t sgemm_n1);

ppl::common::RetCode gemm_fp32(
    const float *a,
    const float *b,
    const float *c,
    float *y,
    float *tmp_buffer,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldy,
    const float alpha,
    const float beta,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3);

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
