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

#include <math.h>

#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode sqrt_fp32(
    const ppl::common::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        float32x4_t src0 = vld1q_f32(x + i + simd_w * 0);
        float32x4_t src1 = vld1q_f32(x + i + simd_w * 1);
        float32x4_t src2 = vld1q_f32(x + i + simd_w * 2);
        float32x4_t src3 = vld1q_f32(x + i + simd_w * 3);

        const float32x4_t dst0 = vsqrt<float32x4_t>(src0);
        const float32x4_t dst1 = vsqrt<float32x4_t>(src1);
        const float32x4_t dst2 = vsqrt<float32x4_t>(src2);
        const float32x4_t dst3 = vsqrt<float32x4_t>(src3);

        vst1q_f32(y + i + simd_w * 0, dst0);
        vst1q_f32(y + i + simd_w * 1, dst1);
        vst1q_f32(y + i + simd_w * 2, dst2);
        vst1q_f32(y + i + simd_w * 3, dst3);
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = sqrt(x[i]);
    }

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode sqrt_fp16(
    const ppl::common::TensorShape *x_shape,
    const __fp16 *x,
    __fp16 *y)
{
    const int64_t n_elem      = x_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        float32x4_t src0 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 0));
        float32x4_t src1 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 1));
        float32x4_t src2 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 2));
        float32x4_t src3 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 3));

        const float32x4_t dst0 = vsqrt<float32x4_t>(src0);
        const float32x4_t dst1 = vsqrt<float32x4_t>(src1);
        const float32x4_t dst2 = vsqrt<float32x4_t>(src2);
        const float32x4_t dst3 = vsqrt<float32x4_t>(src3);

        vst1_f16(y + i + simd_w * 0, vcvt_f16_f32(dst0));
        vst1_f16(y + i + simd_w * 1, vcvt_f16_f32(dst1));
        vst1_f16(y + i + simd_w * 2, vcvt_f16_f32(dst2));
        vst1_f16(y + i + simd_w * 3, vcvt_f16_f32(dst3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = sqrt(x[i]);
    }

    return ppl::common::RC_SUCCESS;
}
#endif

}}}}; // namespace ppl::kernel::arm_server::neon
