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

#ifndef __ST_PPL_KERNEL_X86_COMMON_SSE_TOOLS_H_
#define __ST_PPL_KERNEL_X86_COMMON_SSE_TOOLS_H_

#include <stdint.h>
#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_MSVC

inline __m128 operator+(const __m128 a, const __m128 b) {
    return _mm_add_ps(a, b);
}

inline __m128 operator-(const __m128 a, const __m128 b) {
    return _mm_sub_ps(a, b);
}

inline __m128 operator*(const __m128 a, const __m128 b) {
    return _mm_mul_ps(a, b);
}

inline __m128 operator/(const __m128 a, const __m128 b) {
    return _mm_div_ps(a, b);
}

inline __m128& operator+=(__m128 &a, const __m128 b) {
    return a = _mm_add_ps(a, b);
}

inline __m128& operator-=(__m128 &a, const __m128 b) {
    return a = _mm_sub_ps(a, b);
}

inline __m128& operator*=(__m128 &a, const __m128 b) {
    return a = _mm_mul_ps(a, b);
}

inline __m128& operator/=(__m128 &a, const __m128 b) {
    return a = _mm_div_ps(a, b);
}

inline __m128 operator-(const __m128 a) {
    return _mm_setzero_ps() - a;
}

inline __m128 operator+(const __m128 a) {
    return a;
}

#endif

inline void memset32_sse(void *dst, const int32_t val, const int64_t n32) {
    int64_t __n32 = n32;
    int32_t *__dst = (int32_t*)dst;
    if (__n32 >= 4) {
        __m128 xmm_val = _mm_set1_ps(*reinterpret_cast<const float *>(&val));
        while (__n32 >= 16) {
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 0), xmm_val);
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 4), xmm_val);
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 8), xmm_val);
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 12), xmm_val);
            __dst += 16;
            __n32 -= 16;
        }
        if (__n32 & 8) {
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 0), xmm_val);
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 4), xmm_val);
            __dst += 8;
            __n32 -= 8;
        }
        if (__n32 & 4) {
            _mm_storeu_ps(reinterpret_cast<float *>(__dst + 0), xmm_val);
            __dst += 4;
            __n32 -= 4;
        }
    }
    if (__n32 & 2) {
        __dst[0] = val;
        __dst[1] = val;
        __dst += 2;
        __n32 -= 2;
    }
    if (__n32 & 1) {
        __dst[0] = val;
    }
}

inline void memcpy32_sse(void *dst, const void *src, const int64_t n32) {
    int64_t __n32 = n32;
    float *__dst = (float*)dst;
    const float *__src = (const float*)src;
    while (__n32 >= 16) {
        _mm_storeu_ps(__dst + 0, _mm_loadu_ps(__src + 0));
        _mm_storeu_ps(__dst + 4, _mm_loadu_ps(__src + 4));
        _mm_storeu_ps(__dst + 8, _mm_loadu_ps(__src + 8));
        _mm_storeu_ps(__dst + 12, _mm_loadu_ps(__src + 12));
        __dst += 16;
        __src += 16;
        __n32 -= 16;
    }
    if (__n32 & 8) {
        _mm_storeu_ps(__dst + 0, _mm_loadu_ps(__src + 0));
        _mm_storeu_ps(__dst + 4, _mm_loadu_ps(__src + 4));
        __dst += 8;
        __src += 8;
        __n32 -= 8;
    }
    if (__n32 & 4) {
        _mm_storeu_ps(__dst + 0, _mm_loadu_ps(__src + 0));
        __dst += 4;
        __src += 4;
        __n32 -= 4;
    }
    if (__n32 & 2) {
        __dst[0] = __src[0];
        __dst[1] = __src[1];
        __dst += 2;
        __src += 2;
        __n32 -= 2;
    }
    if (__n32 & 1) {
        __dst[0] = __src[0];
    }
}

}}}; // namespace ppl::kernel::x86

#endif
