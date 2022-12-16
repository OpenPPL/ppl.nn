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

#ifndef __ST_PPL_KERNEL_X86_COMMON_ARITHMETIC_ARITHMETIC_ELTWISE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_ARITHMETIC_ARITHMETIC_ELTWISE_COMMON_H_

#include "ppl/kernel/x86/common/arithmetic/arithmetic_kernel_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT, arithmetic_op_type_t _op>
static ppl::common::RetCode arithmetic_eltwise_common(
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    const int64_t unroll_len  = 8;
    const int64_t length      = dst_shape->CalcElementsIncludingPadding();
    const int64_t unroll_body = round(length, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        dst[i + 0] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 0], src1[i + 0]);
        dst[i + 1] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 1], src1[i + 1]);
        dst[i + 2] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 2], src1[i + 2]);
        dst[i + 3] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 3], src1[i + 3]);
        dst[i + 4] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 4], src1[i + 4]);
        dst[i + 5] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 5], src1[i + 5]);
        dst[i + 6] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 6], src1[i + 6]);
        dst[i + 7] = arithmetic_scalar_kernel_common<eT, _op>(src0[i + 7], src1[i + 7]);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = arithmetic_scalar_kernel_common<eT, _op>(src0[i], src1[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_ARITHMETIC_ARITHMETIC_ELTWISE_COMMON_H_