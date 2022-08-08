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

#ifndef __ST_PPL_KERNEL_X86_FP32_TRANSPOSE_SSE_TRANSPOSE_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_TRANSPOSE_SSE_TRANSPOSE_FP32_SSE_H_

#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

#define TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, tmp0, tmp1, tmp2, tmp3) \
    do {\
        tmp0 = _mm_unpacklo_ps(xmm0, xmm1);\
        tmp1 = _mm_unpackhi_ps(xmm0, xmm1);\
        tmp2 = _mm_unpacklo_ps(xmm2, xmm3);\
        tmp3 = _mm_unpackhi_ps(xmm2, xmm3);\
        xmm0 = _mm_shuffle_ps(tmp0, tmp2, 0x44);\
        xmm1 = _mm_shuffle_ps(tmp0, tmp2, 0xee);\
        xmm2 = _mm_shuffle_ps(tmp1, tmp3, 0x44);\
        xmm3 = _mm_shuffle_ps(tmp1, tmp3, 0xee);\
    } while (false)

inline void transpose_4x4_fp32_sse(
    const float *src,
    const int64_t src_stride,
    const int64_t dst_stride,
    float *dst)
{
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    xmm0 = _mm_loadu_ps(src + 0 * src_stride);
    xmm1 = _mm_loadu_ps(src + 1 * src_stride);
    xmm2 = _mm_loadu_ps(src + 2 * src_stride);
    xmm3 = _mm_loadu_ps(src + 3 * src_stride);

    TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7);

    _mm_storeu_ps(dst + 0 * dst_stride, xmm0);
    _mm_storeu_ps(dst + 1 * dst_stride, xmm1);
    _mm_storeu_ps(dst + 2 * dst_stride, xmm2);
    _mm_storeu_ps(dst + 3 * dst_stride, xmm3);
}

}}}; // namespace ppl::kernel::x86

#endif
