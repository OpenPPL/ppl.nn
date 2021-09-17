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

namespace ppl { namespace kernel { namespace x86 {

typedef int32_t gemm_v_type_t;
class gemm_v_type {
public:
    static const gemm_v_type_t empty = 0;
    static const gemm_v_type_t scalar = 1;
    static const gemm_v_type_t col_vec = 2;
    static const gemm_v_type_t row_vec = 3;
};

typedef int32_t gemm_m_type_t;
class gemm_m_type {
public:
    static const gemm_m_type_t empty = 0;
    static const gemm_m_type_t notrans = 1;
    static const gemm_m_type_t trans = 2;
    static const gemm_m_type_t packed = 3;
};

typedef int32_t gemm_post_t;
class gemm_post {
public:
    static const gemm_post_t none = 0;
    static const gemm_post_t relu = 1;
    static const gemm_post_t relu6 = 2;
};

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
    float *Y);

ppl::common::RetCode gemm_fp32_fma(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
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
    float *C);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_GEMM_H_
