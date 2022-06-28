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

#ifndef __ST_PPL_KERNEL_X86_COMMON_MATH_AVX_H_
#define __ST_PPL_KERNEL_X86_COMMON_MATH_AVX_H_

#include <immintrin.h>

namespace ppl { namespace kernel { namespace x86 {

static inline __m256 _avx_sign_ps(__m256 value) {
    const __m256 zero = _mm256_setzero_ps();
    __m256 positives = _mm256_and_ps(_mm256_cmp_ps(value, zero, _CMP_GT_OQ), _mm256_set1_ps(1.0f));
    __m256 negatives = _mm256_and_ps(_mm256_cmp_ps(value, zero, _CMP_LT_OQ), _mm256_set1_ps(-1.0f));
    return _mm256_or_ps(positives, negatives);
}

}}}; // namespace ppl::kernel::x86

#endif
