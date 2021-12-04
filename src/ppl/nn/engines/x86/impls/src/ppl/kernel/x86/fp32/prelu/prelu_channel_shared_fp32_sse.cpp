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

ppl::common::RetCode prelu_channel_shared_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *slope,
    float *dst)
{

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDimCount() > 1 ? src_shape->GetDim(1) : 1;
    int64_t inner_dim = 1;
    for (uint32_t i = 2; i < src_shape->GetDimCount(); ++i) {
        inner_dim *= src_shape->GetDim(i);
    }

    const int64_t simd_w       = 4;
    const int64_t unroll_inner = 2 * simd_w;
    const __m128 v_zero        = _mm_setzero_ps();

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const float slope_var    = slope[c];
            __m128 slope_vec0        = _mm_set_ps1(slope_var);
            float *base_dst         = dst + n * channels * inner_dim + c * inner_dim;
            const float *base_src   = src + n * channels * inner_dim + c * inner_dim;
            for (int64_t i = 0; i < round(inner_dim, unroll_inner); i += unroll_inner) {

                __m128 v_src0 = _mm_loadu_ps(base_src + simd_w * 0);
                __m128 v_src1 = _mm_loadu_ps(base_src + simd_w * 1);

                __m128 v_ge0 = _mm_max_ps(v_src0, v_zero);
                __m128 v_ge1 = _mm_max_ps(v_src1, v_zero); 

                __m128 v_le0 = _mm_mul_ps(_mm_min_ps(v_src0, v_zero), slope_vec0);
                __m128 v_le1 = _mm_mul_ps(_mm_min_ps(v_src1, v_zero), slope_vec0);

                _mm_storeu_ps(base_dst + simd_w * 0, _mm_add_ps(v_ge0, v_le0));
                _mm_storeu_ps(base_dst + simd_w * 1, _mm_add_ps(v_ge1, v_le1));
                base_dst += unroll_inner;
                base_src += unroll_inner;
            }
            if (round(inner_dim, unroll_inner) < inner_dim) {
                for (int64_t i = 0; i < inner_dim - round(inner_dim, unroll_inner); ++i) {
                    base_dst[i] = base_src[i] >= 0 ? base_src[i] : slope_var * base_src[i];
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;

}

}}}; // namespace ppl::kernel::x86