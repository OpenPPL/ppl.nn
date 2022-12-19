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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode leaky_relu_fp32(
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst)
{
    const int64_t n_elem      = src_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    const float32x4_t v_alpha = vdupq_n_f32(alpha);
    const float32x4_t v_zero  = vdupq_n_f32(0);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        const float32x4_t v_src_0 = vld1q_f32(src + i + simd_w * 0);
        const float32x4_t v_src_1 = vld1q_f32(src + i + simd_w * 1);
        const float32x4_t v_src_2 = vld1q_f32(src + i + simd_w * 2);
        const float32x4_t v_src_3 = vld1q_f32(src + i + simd_w * 3);

        const float32x4_t v_ge_0 = vmaxq_f32(v_src_0, v_zero);
        const float32x4_t v_ge_1 = vmaxq_f32(v_src_1, v_zero);
        const float32x4_t v_ge_2 = vmaxq_f32(v_src_2, v_zero);
        const float32x4_t v_ge_3 = vmaxq_f32(v_src_3, v_zero);

        const float32x4_t v_le_0 = vmulq_f32(vminq_f32(v_src_0, v_zero), v_alpha);
        const float32x4_t v_le_1 = vmulq_f32(vminq_f32(v_src_1, v_zero), v_alpha);
        const float32x4_t v_le_2 = vmulq_f32(vminq_f32(v_src_2, v_zero), v_alpha);
        const float32x4_t v_le_3 = vmulq_f32(vminq_f32(v_src_3, v_zero), v_alpha);

        vst1q_f32(dst + i + simd_w * 0, vaddq_f32(v_ge_0, v_le_0));
        vst1q_f32(dst + i + simd_w * 1, vaddq_f32(v_ge_1, v_le_1));
        vst1q_f32(dst + i + simd_w * 2, vaddq_f32(v_ge_2, v_le_2));
        vst1q_f32(dst + i + simd_w * 3, vaddq_f32(v_ge_3, v_le_3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        dst[i] = src[i] >= 0 ? src[i] : src[i] * alpha;
    }

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode leaky_relu_fp16(
    const ppl::common::TensorShape *src_shape,
    const __fp16 *src,
    const float alpha,
    __fp16 *dst)
{
    const int64_t n_elem      = src_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = 4;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    const float32x4_t v_alpha = vdupq_n_f32(alpha);
    const float32x4_t v_zero  = vdupq_n_f32(0);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        const float32x4_t v_src_0 = vcvt_f32_f16(vld1_f16(src + i + simd_w * 0));
        const float32x4_t v_src_1 = vcvt_f32_f16(vld1_f16(src + i + simd_w * 1));
        const float32x4_t v_src_2 = vcvt_f32_f16(vld1_f16(src + i + simd_w * 2));
        const float32x4_t v_src_3 = vcvt_f32_f16(vld1_f16(src + i + simd_w * 3));

        const float32x4_t v_ge_0 = vmaxq_f32(v_src_0, v_zero);
        const float32x4_t v_ge_1 = vmaxq_f32(v_src_1, v_zero);
        const float32x4_t v_ge_2 = vmaxq_f32(v_src_2, v_zero);
        const float32x4_t v_ge_3 = vmaxq_f32(v_src_3, v_zero);

        const float32x4_t v_le_0 = vmulq_f32(vminq_f32(v_src_0, v_zero), v_alpha);
        const float32x4_t v_le_1 = vmulq_f32(vminq_f32(v_src_1, v_zero), v_alpha);
        const float32x4_t v_le_2 = vmulq_f32(vminq_f32(v_src_2, v_zero), v_alpha);
        const float32x4_t v_le_3 = vmulq_f32(vminq_f32(v_src_3, v_zero), v_alpha);

        vst1_f16(dst + i + simd_w * 0, vcvt_f16_f32(vaddq_f32(v_ge_0, v_le_0)));
        vst1_f16(dst + i + simd_w * 1, vcvt_f16_f32(vaddq_f32(v_ge_1, v_le_1)));
        vst1_f16(dst + i + simd_w * 2, vcvt_f16_f32(vaddq_f32(v_ge_2, v_le_2)));
        vst1_f16(dst + i + simd_w * 3, vcvt_f16_f32(vaddq_f32(v_ge_3, v_le_3)));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        dst[i] = src[i] >= 0 ? src[i] : src[i] * alpha;
    }

    return ppl::common::RC_SUCCESS;
}
#endif

}}}}; // namespace ppl::kernel::arm_server::neon
