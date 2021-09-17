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

#ifndef __ST_PPL_KERNEL_X86_COMMON_AVX512_TOOLS_H_
#define __ST_PPL_KERNEL_X86_COMMON_AVX512_TOOLS_H_

#include <stdint.h>
#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/common/sse_tools.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_MSVC

inline __m512 operator+(const __m512 a, const __m512 b) {
    return _mm512_add_ps(a, b);
}

inline __m512 operator-(const __m512 a, const __m512 b) {
    return _mm512_sub_ps(a, b);
}

inline __m512 operator*(const __m512 a, const __m512 b) {
    return _mm512_mul_ps(a, b);
}

inline __m512 operator/(const __m512 a, const __m512 b) {
    return _mm512_div_ps(a, b);
}

inline __m512& operator+=(__m512 &a, const __m512 b) {
    return a = _mm512_add_ps(a, b);
}

inline __m512& operator-=(__m512 &a, const __m512 b) {
    return a = _mm512_sub_ps(a, b);
}

inline __m512& operator*=(__m512 &a, const __m512 b) {
    return a = _mm512_mul_ps(a, b);
}

inline __m512& operator/=(__m512 &a, const __m512 b) {
    return a = _mm512_div_ps(a, b);
}

inline __m512 operator-(const __m512 a) {
    return _mm512_setzero_ps() - a;
}

inline __m512 operator+(const __m512 a) {
    return a;
}

#endif

}}}; // namespace ppl::kernel::x86

#endif
