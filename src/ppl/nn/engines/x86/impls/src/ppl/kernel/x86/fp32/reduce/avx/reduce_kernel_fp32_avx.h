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

#ifndef __ST_PPL_KERNEL_X86_FP32_REDUCE_AVX_REDUCE_KERNEL_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_FP32_REDUCE_AVX_REDUCE_KERNEL_FP32_AVX_H_

#include <string.h>
#include <float.h>
#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/reduce/reduce_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <reduce_op_type_t _op>
inline float reduce_init_val_fp32(void)
{
    return 0;
}

template <>
inline float reduce_init_val_fp32<REDUCE_MAX>(void)
{
    return -FLT_MAX;
}

template <>
inline float reduce_init_val_fp32<REDUCE_MIN>(void)
{
    return FLT_MAX;
}

template <reduce_op_type_t _op>
static void reduce_preprocess_fp32_avx(
    float *dst,
    int64_t len)
{
    const float init_val    = reduce_init_val_fp32<_op>();
    const __m256 v_init_val = _mm256_set1_ps(init_val);

    const int64_t simd_w      = 8;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(len, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        _mm256_storeu_ps(dst + i + simd_w * 0, v_init_val);
        _mm256_storeu_ps(dst + i + simd_w * 1, v_init_val);
        _mm256_storeu_ps(dst + i + simd_w * 2, v_init_val);
        _mm256_storeu_ps(dst + i + simd_w * 3, v_init_val);
    }
    for (int64_t i = unroll_body; i < len; i++) {
        dst[i] = init_val;
    }
}

template <reduce_op_type_t _op>
inline float reduce_scalar_kernel_fp32(float a, float r);

template <>
inline float reduce_scalar_kernel_fp32<REDUCE_MEAN>(float a, float r)
{
    return a + r;
}

template <>
inline float reduce_scalar_kernel_fp32<REDUCE_MAX>(float a, float r)
{
    return a > r ? a : r;
}

template <>
inline float reduce_scalar_kernel_fp32<REDUCE_MIN>(float a, float r)
{
    return a < r ? a : r;
}

template <>
inline float reduce_scalar_kernel_fp32<REDUCE_SUM>(float a, float r)
{
    return a + r;
}

template <reduce_op_type_t _op>
inline __m256 reduce_vector_kernel_fp32_avx(__m256 a, __m256 r);

template <>
inline __m256 reduce_vector_kernel_fp32_avx<REDUCE_MEAN>(__m256 a, __m256 r)
{
    return _mm256_add_ps(a, r);
}

template <>
inline __m256 reduce_vector_kernel_fp32_avx<REDUCE_MAX>(__m256 a, __m256 r)
{
    return _mm256_max_ps(a, r);
}

template <>
inline __m256 reduce_vector_kernel_fp32_avx<REDUCE_MIN>(__m256 a, __m256 r)
{
    return _mm256_min_ps(a, r);
}

template <>
inline __m256 reduce_vector_kernel_fp32_avx<REDUCE_SUM>(__m256 a, __m256 r)
{
    return _mm256_add_ps(a, r);
}

template <reduce_op_type_t _op>
inline float reduce_vector_all_lanes_kernel_fp32_avx(__m256 v)
{
    float tmp[8];
    _mm256_storeu_ps(tmp, v);
    tmp[0] = reduce_scalar_kernel_fp32<_op>(tmp[0], tmp[1]);
    tmp[2] = reduce_scalar_kernel_fp32<_op>(tmp[2], tmp[3]);
    tmp[4] = reduce_scalar_kernel_fp32<_op>(tmp[4], tmp[5]);
    tmp[6] = reduce_scalar_kernel_fp32<_op>(tmp[6], tmp[7]);
    tmp[0] = reduce_scalar_kernel_fp32<_op>(tmp[0], tmp[2]);
    tmp[4] = reduce_scalar_kernel_fp32<_op>(tmp[4], tmp[6]);
    tmp[0] = reduce_scalar_kernel_fp32<_op>(tmp[0], tmp[4]);
    return tmp[0];
}

template <reduce_op_type_t _op>
static void reduce_postprocess_fp32_avx(
    float *dst,
    int64_t len,
    float div)
{
    if (_op == REDUCE_MEAN) {
        const float rdiv    = 1.0f / div;
        const __m256 v_rdiv = _mm256_set1_ps(rdiv);

        const int64_t simd_w      = 8;
        const int64_t unroll_len  = simd_w * 4;
        const int64_t unroll_body = round(len, unroll_len);

        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            __m256 v_dst_0 = _mm256_loadu_ps(dst + i + simd_w * 0);
            __m256 v_dst_1 = _mm256_loadu_ps(dst + i + simd_w * 1);
            __m256 v_dst_2 = _mm256_loadu_ps(dst + i + simd_w * 2);
            __m256 v_dst_3 = _mm256_loadu_ps(dst + i + simd_w * 3);

            v_dst_0 = _mm256_mul_ps(v_dst_0, v_rdiv);
            v_dst_1 = _mm256_mul_ps(v_dst_1, v_rdiv);
            v_dst_2 = _mm256_mul_ps(v_dst_2, v_rdiv);
            v_dst_3 = _mm256_mul_ps(v_dst_3, v_rdiv);

            _mm256_storeu_ps(dst + i + simd_w * 0, v_dst_0);
            _mm256_storeu_ps(dst + i + simd_w * 1, v_dst_1);
            _mm256_storeu_ps(dst + i + simd_w * 2, v_dst_2);
            _mm256_storeu_ps(dst + i + simd_w * 3, v_dst_3);
        }
        for (int64_t i = unroll_body; i < len; i++) {
            dst[i] *= rdiv;
        }
    }
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_REDUCE_AVX_REDUCE_FP32_AVX_H_