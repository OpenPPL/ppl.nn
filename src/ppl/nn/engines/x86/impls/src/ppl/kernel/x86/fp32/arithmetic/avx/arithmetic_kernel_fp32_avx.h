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

#ifndef __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_KERNEL_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_KERNEL_FP32_AVX_H_

#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/arithmetic/arithmetic_common.h"
#include "ppl/kernel/x86/common/threading_tools.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op>
inline float arithmetic_scalar_kernel_fp32_avx(float a, float b);

template <>
inline float arithmetic_scalar_kernel_fp32_avx<ARITHMETIC_ADD>(float a, float b)
{
    return a + b;
}
template <>
inline float arithmetic_scalar_kernel_fp32_avx<ARITHMETIC_SUB>(float a, float b)
{
    return a - b;
}
template <>
inline float arithmetic_scalar_kernel_fp32_avx<ARITHMETIC_MUL>(float a, float b)
{
    return a * b;
}
template <>
inline float arithmetic_scalar_kernel_fp32_avx<ARITHMETIC_DIV>(float a, float b)
{
    return a / b;
}

template <arithmetic_op_type_t _op>
inline __m256 arithmetic_vector_kernel_fp32_avx(__m256 a, __m256 b);

template <>
inline __m256 arithmetic_vector_kernel_fp32_avx<ARITHMETIC_ADD>(__m256 a, __m256 b)
{
    return _mm256_add_ps(a, b);
}
template <>
inline __m256 arithmetic_vector_kernel_fp32_avx<ARITHMETIC_SUB>(__m256 a, __m256 b)
{
    return _mm256_sub_ps(a, b);
}
template <>
inline __m256 arithmetic_vector_kernel_fp32_avx<ARITHMETIC_MUL>(__m256 a, __m256 b)
{
    return _mm256_mul_ps(a, b);
}
template <>
inline __m256 arithmetic_vector_kernel_fp32_avx<ARITHMETIC_DIV>(__m256 a, __m256 b)
{
    return _mm256_div_ps(a, b);
}

struct parallel_block {
    int64_t id;
    int64_t start[PPL_X86_TENSOR_MAX_DIMS()];
    int64_t end[PPL_X86_TENSOR_MAX_DIMS()];
    int64_t idx[PPL_X86_TENSOR_MAX_DIMS()];
};

inline void pad_shape(
    const ppl::nn::TensorShape *shape,
    const int64_t padded_dim_count,
    int64_t *padded_shape)
{
    const int64_t dim_diff = padded_dim_count - shape->GetRealDimCount();
    for (int64_t i = 0; i < dim_diff; i++) {
        padded_shape[i] = 1;
    }
    for (int64_t i = dim_diff; i < padded_dim_count; i++) {
        padded_shape[i] = shape->GetDim(i - dim_diff);
    }
}

inline void idx2dims(
    const int64_t idx,
    const int64_t *shape,
    const int64_t dim_count,
    int64_t *dims)
{
    int64_t _idx = idx;
    for (int64_t i = dim_count - 1; i >= 0; i--) {
        dims[i] = _idx % shape[i];
        _idx /= shape[i];
    }
}

inline bool is_first_dim(parallel_block *block, const int64_t dim_idx)
{
    bool is_first = true;
    for (int64_t i = 0; i < dim_idx; i++) {
        if (block->idx[i] != block->start[i]) {
            is_first = false;
            break;
        }
    }
    return is_first;
}

inline bool is_last_dim(parallel_block *block, const int64_t dim_idx)
{
    bool is_last = true;
    for (int64_t i = 0; i < dim_idx; i++) {
        if (block->idx[i] != block->end[i]) {
            is_last = false;
            break;
        }
    }
    return is_last;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_KERNEL_FP32_AVX_H_