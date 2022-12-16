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

#include "ppl/kernel/x86/common/internal_include.h"
#include <immintrin.h>

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode not_bool_avx(
    const ppl::common::TensorShape *x_shape,
    const uint8_t *x,
    uint8_t *y)
{
    const int64_t n_elem           = x_shape->CalcElementsIncludingPadding();
    const int64_t n_elem_fp32      = n_elem / 4;
    const int64_t simd_w_fp32      = 8;
    const int64_t unroll_len_fp32  = simd_w_fp32 * 4;
    const int64_t unroll_body_fp32 = round(n_elem_fp32, unroll_len_fp32);
    const int64_t unroll_body      = unroll_body_fp32 * 4;

    const float *src = (const float *)x;
    float *dst       = (float *)y;

    uint32_t maxval[8] = {0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101};
    float *maxvalf     = (float *)maxval;
    __m256 mm_max      = _mm256_loadu_ps(maxvalf);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body_fp32; i += unroll_len_fp32) {
        __m256 mm_var0 = _mm256_loadu_ps(src + i + simd_w_fp32 * 0);
        __m256 mm_var1 = _mm256_loadu_ps(src + i + simd_w_fp32 * 1);
        __m256 mm_var2 = _mm256_loadu_ps(src + i + simd_w_fp32 * 2);
        __m256 mm_var3 = _mm256_loadu_ps(src + i + simd_w_fp32 * 3);

        mm_var0 = _mm256_xor_ps(mm_var0, mm_max);
        mm_var1 = _mm256_xor_ps(mm_var1, mm_max);
        mm_var2 = _mm256_xor_ps(mm_var2, mm_max);
        mm_var3 = _mm256_xor_ps(mm_var3, mm_max);

        _mm256_storeu_ps(dst + i + simd_w_fp32 * 0, mm_var0);
        _mm256_storeu_ps(dst + i + simd_w_fp32 * 1, mm_var1);
        _mm256_storeu_ps(dst + i + simd_w_fp32 * 2, mm_var2);
        _mm256_storeu_ps(dst + i + simd_w_fp32 * 3, mm_var3);
    }

    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = x[i] ^ 0x01;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86