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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_ELTWISE_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_ELTWISE_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, arithmetic_op_type_t op_type, bool fuse_relu>
static ppl::common::RetCode arithmetic_eltwise_common(
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t length      = dst_shape->CalcElementsIncludingPadding();
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    const vecType v_zero = vdup_n<eT, eN>(0);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        vecType v_src0_0 = vld<eT, eN>(src0 + i + simd_w * 0);
        vecType v_src0_1 = vld<eT, eN>(src0 + i + simd_w * 1);
        vecType v_src0_2 = vld<eT, eN>(src0 + i + simd_w * 2);
        vecType v_src0_3 = vld<eT, eN>(src0 + i + simd_w * 3);

        vecType v_src1_0 = vld<eT, eN>(src1 + i + simd_w * 0);
        vecType v_src1_1 = vld<eT, eN>(src1 + i + simd_w * 1);
        vecType v_src1_2 = vld<eT, eN>(src1 + i + simd_w * 2);
        vecType v_src1_3 = vld<eT, eN>(src1 + i + simd_w * 3);

        vecType v_dst_0 = arithmetic_vector_kernel<vecType, op_type>(v_src0_0, v_src1_0);
        vecType v_dst_1 = arithmetic_vector_kernel<vecType, op_type>(v_src0_1, v_src1_1);
        vecType v_dst_2 = arithmetic_vector_kernel<vecType, op_type>(v_src0_2, v_src1_2);
        vecType v_dst_3 = arithmetic_vector_kernel<vecType, op_type>(v_src0_3, v_src1_3);

        if (fuse_relu) {
            v_dst_0 = vmax<vecType>(v_dst_0, v_zero);
            v_dst_1 = vmax<vecType>(v_dst_1, v_zero);
            v_dst_2 = vmax<vecType>(v_dst_2, v_zero);
            v_dst_3 = vmax<vecType>(v_dst_3, v_zero);
        }

        vst<eT, eN>(dst + i + simd_w * 0, v_dst_0);
        vst<eT, eN>(dst + i + simd_w * 1, v_dst_1);
        vst<eT, eN>(dst + i + simd_w * 2, v_dst_2);
        vst<eT, eN>(dst + i + simd_w * 3, v_dst_3);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        eT dst_val = arithmetic_scalar_kernel<eT, op_type>(src0[i], src1[i]);
        if (fuse_relu) {
            dst_val = max(dst_val, (eT)0);
        }
        dst[i] = dst_val;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_ELTWISE_COMMON_H_