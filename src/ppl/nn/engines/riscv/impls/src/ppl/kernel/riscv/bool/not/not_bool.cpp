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

#include <riscv-vector.h>
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode not_bool(
    const ppl::common::TensorShape* src_shape,
    const uint8_t* src,
    uint8_t* dst)
{
    const int64_t n_elem           = src_shape->CalcElementsIncludingPadding();
    const int64_t n_elem_fp32      = n_elem / 4;
    const int64_t simd_w_fp32      = 4;
    const int64_t unroll_len_fp32  = simd_w_fp32 * 4;
    const int64_t unroll_body_fp32 = round(n_elem_fp32, unroll_len_fp32);
    const int64_t unroll_body      = unroll_body_fp32 * 4;

    const uint32_t* src_ = (const uint32_t*)src;
    uint32_t* dst_       = (uint32_t*)dst;

    auto vl             = vsetvli(4, RVV_E32, RVV_M1);
    uint32_t xor_val[4] = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    uint32xm1_t xor_var = vlev_uint32xm1(xor_val, vl);

    for (int64_t i = 0; i < unroll_body_fp32; i += unroll_len_fp32) {
        uint32xm1_t var_0 = vlev_uint32xm1(src_ + i + simd_w_fp32 * 0, vl);
        uint32xm1_t var_1 = vlev_uint32xm1(src_ + i + simd_w_fp32 * 1, vl);
        uint32xm1_t var_2 = vlev_uint32xm1(src_ + i + simd_w_fp32 * 2, vl);
        uint32xm1_t var_3 = vlev_uint32xm1(src_ + i + simd_w_fp32 * 3, vl);

        var_0 = vxorvv_uint32xm1(var_0, xor_var, vl);
        var_1 = vxorvv_uint32xm1(var_1, xor_var, vl);
        var_2 = vxorvv_uint32xm1(var_2, xor_var, vl);
        var_3 = vxorvv_uint32xm1(var_3, xor_var, vl);

        vsev_uint32xm1(dst_ + i + simd_w_fp32 * 0, var_0, vl);
        vsev_uint32xm1(dst_ + i + simd_w_fp32 * 1, var_1, vl);
        vsev_uint32xm1(dst_ + i + simd_w_fp32 * 2, var_2, vl);
        vsev_uint32xm1(dst_ + i + simd_w_fp32 * 3, var_3, vl);
    }
    for (int64_t i = unroll_body; i < n_elem; i++) {
        dst[i] = src[i] ^ 0x01;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv