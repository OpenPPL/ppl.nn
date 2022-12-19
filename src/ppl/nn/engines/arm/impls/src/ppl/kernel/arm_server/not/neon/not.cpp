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


#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode not_bool(
    const ppl::common::TensorShape *x_shape,
    const uint8_t *x,   
    uint8_t* y)
{
    const int64_t n_elem           = x_shape->CalcElementsIncludingPadding();
    const int64_t n_elem_fp32      = n_elem / 4;
    const int64_t simd_w_fp32      = 4;
    const int64_t unroll_len_fp32  = simd_w_fp32 * 4;
    const int64_t unroll_body_fp32 = round(n_elem_fp32, unroll_len_fp32);
    const int64_t unroll_body      = unroll_body_fp32 * 4;

    const uint32_t* src = (const uint32_t*) x;
    uint32_t* dst     = (uint32_t*) y;

    uint32_t maxval[4] = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    uint32x4_t mm_max = vld1q_u32(maxval);
    

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body_fp32; i += unroll_len_fp32) {
        uint32x4_t mm_var0 = vld1q_u32(src + i + simd_w_fp32 * 0);
        uint32x4_t mm_var1 = vld1q_u32(src + i + simd_w_fp32 * 1);
        uint32x4_t mm_var2 = vld1q_u32(src + i + simd_w_fp32 * 2);
        uint32x4_t mm_var3 = vld1q_u32(src + i + simd_w_fp32 * 3);

        mm_var0 = veorq_u32(mm_var0, mm_max);
        mm_var1 = veorq_u32(mm_var1, mm_max);
        mm_var2 = veorq_u32(mm_var2, mm_max);
        mm_var3 = veorq_u32(mm_var3, mm_max);

        vst1q_u32(dst + i + simd_w_fp32 * 0, mm_var0);
        vst1q_u32(dst + i + simd_w_fp32 * 1, mm_var1);
        vst1q_u32(dst + i + simd_w_fp32 * 2, mm_var2);
        vst1q_u32(dst + i + simd_w_fp32 * 3, mm_var3);

    }

    for (int64_t i = unroll_body; i < n_elem; i++)
    {
        y[i] = x[i] ^ 0x01;
    }
    
    return ppl::common::RC_SUCCESS;
}

}}}}; // namespace ppl::kernel::arm_server::neon
