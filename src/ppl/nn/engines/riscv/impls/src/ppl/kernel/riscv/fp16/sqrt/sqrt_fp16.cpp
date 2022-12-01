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
#include <math.h>
#include "ppl/kernel/riscv/common/internal_include.h"

#define C_BLK() 8

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode sqrt_fp16(
    const ppl::common::TensorShape* src_shape,
    const __fp16* src,
    __fp16* dst)
{
    const int64_t n_elem      = src_shape->CalcElementsIncludingPadding();
    const int64_t unroll_n    = C_BLK();
    const int64_t unroll_body = round(n_elem, unroll_n);

    const auto vl16 = vsetvli(8, RVV_E16, RVV_M1);
    const auto vl32 = vsetvli(4, RVV_E32, RVV_M2);

    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        // handle low 4 elements of float16
        float16xm1_t v_src0_fp16_l = vlev_float16xm1(src + i + 0 * 4, vl16);
        float32xm2_t v_src0_fp32_l = vfwcvtffv_float32xm2_float16xm1(v_src0_fp16_l, vl16);
        float32xm2_t v_dst0_fp32_l = vfsqrtv_float32xm2(v_src0_fp32_l, vl32);
        float16xm1_t v_dst0_fp16_l = vfncvtffv_float16xm1_float32xm2(v_dst0_fp32_l, vl32);
        vsev_float16xm1(dst + i + 0 * 4, v_dst0_fp16_l, vl16);
        // handle high 4 elements of float16
        float16xm1_t v_src0_fp16_h = vlev_float16xm1(src + i + 1 * 4, vl16);
        float32xm2_t v_src0_fp32_h = vfwcvtffv_float32xm2_float16xm1(v_src0_fp16_h, vl16);
        float32xm2_t v_dst0_fp32_h = vfsqrtv_float32xm2(v_src0_fp32_h, vl32);
        float16xm1_t v_dst0_fp16_h = vfncvtffv_float16xm1_float32xm2(v_dst0_fp32_h, vl32);
        vsev_float16xm1(dst + i + 1 * 4, v_dst0_fp16_h, vl16);
    }
    for (int64_t i = unroll_body; i < n_elem; i++) {
        dst[i] = sqrt((float)src[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv