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

#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode leaky_relu_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst)
{
    const uint64_t simd_w      = 4;
    const uint64_t unroll_len  = simd_w * 4;
    const uint64_t unroll_body = round(src_shape->GetElementsIncludingPadding(), unroll_len);
    const __m128 v_alpha       = _mm_set_ps1(alpha);
    const __m128 v_zero        = _mm_setzero_ps();
    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
        __m128 v_src0 = _mm_loadu_ps(src + i + 0 * simd_w);
        __m128 v_src1 = _mm_loadu_ps(src + i + 1 * simd_w);
        __m128 v_src2 = _mm_loadu_ps(src + i + 2 * simd_w);
        __m128 v_src3 = _mm_loadu_ps(src + i + 3 * simd_w);

        __m128 v_ge0 = _mm_max_ps(v_src0, v_zero);
        __m128 v_ge1 = _mm_max_ps(v_src1, v_zero);
        __m128 v_ge2 = _mm_max_ps(v_src2, v_zero);
        __m128 v_ge3 = _mm_max_ps(v_src3, v_zero);

        __m128 v_le0 = _mm_mul_ps(_mm_min_ps(v_src0, v_zero), v_alpha);
        __m128 v_le1 = _mm_mul_ps(_mm_min_ps(v_src1, v_zero), v_alpha);
        __m128 v_le2 = _mm_mul_ps(_mm_min_ps(v_src2, v_zero), v_alpha);
        __m128 v_le3 = _mm_mul_ps(_mm_min_ps(v_src3, v_zero), v_alpha);

        _mm_storeu_ps(dst + i + 0 * simd_w, _mm_add_ps(v_ge0, v_le0));
        _mm_storeu_ps(dst + i + 1 * simd_w, _mm_add_ps(v_ge1, v_le1));
        _mm_storeu_ps(dst + i + 2 * simd_w, _mm_add_ps(v_ge2, v_le2));
        _mm_storeu_ps(dst + i + 3 * simd_w, _mm_add_ps(v_ge3, v_le3));
    }
    for (uint64_t i = unroll_body; i < src_shape->GetElementsIncludingPadding(); i++) {
        dst[i] = src[i] >= 0 ? src[i] : alpha * src[i];
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86