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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_KERNEL_FP32_SGEMM_NDARRAY_KERNEL_H_
#define __ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_KERNEL_FP32_SGEMM_NDARRAY_KERNEL_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

typedef void (*sgemm_ndarray_kernel_func_t)(
    const float* A, 
    const float* B, 
    const int64_t K, 
    const int64_t lda, 
    const int64_t ldb, 
    const int64_t ldc, 
    float* C);

extern const sgemm_ndarray_kernel_func_t sgemm_ndarray_kernel_tn_max8x12_func_table[2][2][2][8][3];

}}}}

#endif  // !__ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_KERNEL_FP32_SGEMM_NDARRAY_KERNEL_H_
