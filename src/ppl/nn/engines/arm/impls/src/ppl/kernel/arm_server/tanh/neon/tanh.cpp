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
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/math_neon.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode tanh_fp32(
    const ppl::common::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        const float32x4_t v_src_0 = vld1q_f32(x + i + simd_w * 0);
        const float32x4_t v_src_1 = vld1q_f32(x + i + simd_w * 1);
        const float32x4_t v_src_2 = vld1q_f32(x + i + simd_w * 2);
        const float32x4_t v_src_3 = vld1q_f32(x + i + simd_w * 3);

        vst1q_f32(y + i + simd_w * 0, v_tanh_f32(v_src_0));
        vst1q_f32(y + i + simd_w * 1, v_tanh_f32(v_src_1));
        vst1q_f32(y + i + simd_w * 2, v_tanh_f32(v_src_2));
        vst1q_f32(y + i + simd_w * 3, v_tanh_f32(v_src_3));
    }

    int64_t i = unroll_body;
    for (; i + simd_w <= n_elem; i += simd_w) {
        const float32x4_t v_src = vld1q_f32(x + i);
        vst1q_f32(y + i, v_tanh_f32(v_src));
    }
    for (; i < n_elem; ++i) {
        const float32x4_t v_src = vdupq_n_f32(x[i]);
        y[i]                    = vgetq_lane_f32(v_tanh_f32(v_src), 0);
    }

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode tanh_fp16(
    const ppl::common::TensorShape *x_shape,
    const __fp16 *x,
    __fp16 *y)
{
    const int64_t n_elem      = x_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        const float32x4_t v_src_0 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 0));
        const float32x4_t v_src_1 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 1));
        const float32x4_t v_src_2 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 2));
        const float32x4_t v_src_3 = vcvt_f32_f16(vld1_f16(x + i + simd_w * 3));

        vst1_f16(y + i + simd_w * 0, vcvt_f16_f32(v_tanh_f32(v_src_0)));
        vst1_f16(y + i + simd_w * 1, vcvt_f16_f32(v_tanh_f32(v_src_1)));
        vst1_f16(y + i + simd_w * 2, vcvt_f16_f32(v_tanh_f32(v_src_2)));
        vst1_f16(y + i + simd_w * 3, vcvt_f16_f32(v_tanh_f32(v_src_3)));
    }

    int64_t i = unroll_body;
    for (; i + simd_w <= n_elem; i += simd_w) {
        const float32x4_t v_src = vcvt_f32_f16(vld1_f16(x + i));
        vst1_f16(y + i, vcvt_f16_f32(v_tanh_f32(v_src)));
    }
    for (; i < n_elem; ++i) {
        const float32x4_t v_src = vdupq_n_f32((float)x[i]);
        y[i]                    = (__fp16)vgetq_lane_f32(v_tanh_f32(v_src), 0);
    }

    return ppl::common::RC_SUCCESS;
}
#endif

}}}}; // namespace ppl::kernel::arm_server::neon
