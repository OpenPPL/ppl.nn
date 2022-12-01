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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_ELTWISE_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_ELTWISE_COMMON_H_

#include "ppl/kernel/riscv/common/relation/relation_common.h"

namespace ppl { namespace kernel { namespace riscv {

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_eltwise_scalar_common(
    const ppl::common::TensorShape *dst_shape,
    const T *src0,
    const T *src1,
    uint8_t *dst)
{
    const int64_t length = dst_shape->CalcElementsIncludingPadding();
    for (int64_t i = 0; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(src0[i], src1[i]);
    }

    return ppl::common::RC_SUCCESS;
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_eltwise_common(
    const ppl::common::TensorShape *dst_shape,
    const T *src0,
    const T *src1,
    uint8_t *dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    const int64_t simd_w      = c_blk;
    const int64_t length      = dst_shape->CalcElementsIncludingPadding();
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        register_v<T, vlen> v_src0_0 = vlev<T, vlen>(src0 + i + simd_w * 0, vl);
        register_v<T, vlen> v_src0_1 = vlev<T, vlen>(src0 + i + simd_w * 1, vl);
        register_v<T, vlen> v_src0_2 = vlev<T, vlen>(src0 + i + simd_w * 2, vl);
        register_v<T, vlen> v_src0_3 = vlev<T, vlen>(src0 + i + simd_w * 3, vl);

        register_v<T, vlen> v_src1_0 = vlev<T, vlen>(src1 + i + simd_w * 0, vl);
        register_v<T, vlen> v_src1_1 = vlev<T, vlen>(src1 + i + simd_w * 1, vl);
        register_v<T, vlen> v_src1_2 = vlev<T, vlen>(src1 + i + simd_w * 2, vl);
        register_v<T, vlen> v_src1_3 = vlev<T, vlen>(src1 + i + simd_w * 3, vl);

        register_ve<T, vlen> v_dst_0 = vrelation_vv<op, T, vlen>(v_src0_0, v_src1_0, vl);
        register_ve<T, vlen> v_dst_1 = vrelation_vv<op, T, vlen>(v_src0_1, v_src1_1, vl);
        register_ve<T, vlen> v_dst_2 = vrelation_vv<op, T, vlen>(v_src0_2, v_src1_2, vl);
        register_ve<T, vlen> v_dst_3 = vrelation_vv<op, T, vlen>(v_src0_3, v_src1_3, vl);

        pack_four<T, vlen>(v_dst_0, v_dst_1, v_dst_2, v_dst_3, v_mask, dst + i);
    }

    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(src0[i], src1[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif
