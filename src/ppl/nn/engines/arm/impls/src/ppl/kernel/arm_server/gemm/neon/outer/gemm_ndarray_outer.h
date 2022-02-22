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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_OUTER_GEMM_NDARRAY_OUTER_H_
#define __ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_OUTER_GEMM_NDARRAY_OUTER_H_

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/gemm/neon/gemm.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
ppl::common::RetCode gemm_ndarray_common_outer(
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
    eT* Y);

}}}} // namespace ppl::kernel::arm_server::neon

#endif // !__ST_PPL_KERNEL_ARM_SERVER_GEMM_NEON_OUTER_GEMM_NDARRAY_OUTER_H_
