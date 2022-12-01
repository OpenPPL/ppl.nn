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

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode sqrt_fp32(
    const ppl::common::TensorShape* src_shape,
    const float* src,
    float* dst)
{
    const int64_t n_elem      = src_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    const auto vl = vsetvli(4, RVV_E32, RVV_M1);

    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        float32xm1_t src0 = vlev_float32xm1(src + i + simd_w * 0, vl);
        float32xm1_t src1 = vlev_float32xm1(src + i + simd_w * 1, vl);
        float32xm1_t src2 = vlev_float32xm1(src + i + simd_w * 2, vl);
        float32xm1_t src3 = vlev_float32xm1(src + i + simd_w * 3, vl);

        float32xm1_t dst0 = vfsqrtv_float32xm1(src0, vl);
        float32xm1_t dst1 = vfsqrtv_float32xm1(src1, vl);
        float32xm1_t dst2 = vfsqrtv_float32xm1(src2, vl);
        float32xm1_t dst3 = vfsqrtv_float32xm1(src3, vl);

        vsev_float32xm1(dst + i + simd_w * 0, dst0, vl);
        vsev_float32xm1(dst + i + simd_w * 1, dst1, vl);
        vsev_float32xm1(dst + i + simd_w * 2, dst2, vl);
        vsev_float32xm1(dst + i + simd_w * 3, dst3, vl);
    }
    for (int64_t i = unroll_body; i < n_elem; i++) {
        dst[i] = sqrt(src[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv