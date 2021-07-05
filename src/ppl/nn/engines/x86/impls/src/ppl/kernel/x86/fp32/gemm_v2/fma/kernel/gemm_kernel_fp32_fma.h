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

#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

typedef void (*gemm_kernel_fp32_fma_func_type_t)(const float*, const float*, const int32_t, const int32_t, const int32_t, const int32_t, float*);

// 6x16 kernels
extern const gemm_kernel_fp32_fma_func_type_t gemm_kernel_max6x16_fp32_fma_func_tab[7][3];

void gemm_kernel_6x16_fp32_fma(
    const float* A,
    const float* B,
    const int32_t k_len,
    const int32_t lda,
    const int32_t ldb,
    const int32_t ldc,
    float* C);

}}} // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_KERNEL_FP32_FMA_H_