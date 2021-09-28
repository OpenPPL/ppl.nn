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

#ifndef __ST_PPL_KERNEL_X86_FP32_RELATION_AVX_RELATION_FP32_AVX_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_RELATION_AVX_RELATION_FP32_AVX_COMMON_H_

#include <immintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/relation/relation_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <relation_op_type_t _op>
inline uint8_t relation_scalar_kernel_fp32(float a, float b);

template <>
inline uint8_t relation_scalar_kernel_fp32<RELATION_GREATER>(float a, float b)
{
    return a > b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_fp32<RELATION_GREATER_OR_EQUAL>(float a, float b)
{
    return a >= b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_fp32<RELATION_LESS>(float a, float b)
{
    return a < b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_fp32<RELATION_LESS_OR_EQUAL>(float a, float b)
{
    return a <= b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_fp32<RELATION_EQUAL>(float a, float b)
{
    return a == b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_fp32<RELATION_NOT_EQUAL>(float a, float b)
{
    return a != b ? 1 : 0;
}

template <relation_op_type_t _op>
inline __m256 relation_vector_kernel_fp32_avx(__m256 va, __m256 vb);

template <>
inline __m256 relation_vector_kernel_fp32_avx<RELATION_GREATER>(__m256 va, __m256 vb)
{
    return _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
}
template <>
inline __m256 relation_vector_kernel_fp32_avx<RELATION_GREATER_OR_EQUAL>(__m256 va, __m256 vb)
{
    return _mm256_cmp_ps(va, vb, _CMP_GE_OQ);
}
template <>
inline __m256 relation_vector_kernel_fp32_avx<RELATION_LESS>(__m256 va, __m256 vb)
{
    return _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
}
template <>
inline __m256 relation_vector_kernel_fp32_avx<RELATION_LESS_OR_EQUAL>(__m256 va, __m256 vb)
{
    return _mm256_cmp_ps(va, vb, _CMP_LE_OQ);
}
template <>
inline __m256 relation_vector_kernel_fp32_avx<RELATION_EQUAL>(__m256 va, __m256 vb)
{
    return _mm256_cmp_ps(va, vb, _CMP_EQ_OQ);
}
template <>
inline __m256 relation_vector_kernel_fp32_avx<RELATION_NOT_EQUAL>(__m256 va, __m256 vb)
{
    return _mm256_cmp_ps(va, vb, _CMP_NEQ_OQ);
}

#define PACK_4UINT32X8_TO_UINT32X32_AVX(v0, v1, v2, v3, dst) do {\
    uint32_t tmp[32];\
    uint8_t *tmp_dst = dst;\
    _mm256_storeu_si256((__m256i*)(tmp + 0), _mm256_castps_si256(v0));\
    _mm256_storeu_si256((__m256i*)(tmp + 8), _mm256_castps_si256(v1));\
    _mm256_storeu_si256((__m256i*)(tmp + 16), _mm256_castps_si256(v2));\
    _mm256_storeu_si256((__m256i*)(tmp + 24), _mm256_castps_si256(v3));\
    for (uint64_t _ii = 0; _ii < 32; _ii++) {\
        tmp_dst[_ii] = tmp[_ii] & 0x00000001;\
    }\
} while (false)

template <relation_op_type_t _op>
ppl::common::RetCode relation_eltwise_binary_op_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst)
{
    const uint64_t simd_w      = 8;
    const uint64_t unroll_len  = simd_w * 4;
    const uint64_t length = dst_shape->GetElementsIncludingPadding();
    const uint64_t unroll_body = round(length, unroll_len);
    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
        __m256 mm0 = _mm256_loadu_ps(src0 + i + 0 * simd_w);
        __m256 mm1 = _mm256_loadu_ps(src0 + i + 1 * simd_w);
        __m256 mm2 = _mm256_loadu_ps(src0 + i + 2 * simd_w);
        __m256 mm3 = _mm256_loadu_ps(src0 + i + 3 * simd_w);
        mm0        = relation_vector_kernel_fp32_avx<_op>(mm0, _mm256_loadu_ps(src1 + i + 0 * simd_w));
        mm1        = relation_vector_kernel_fp32_avx<_op>(mm1, _mm256_loadu_ps(src1 + i + 1 * simd_w));
        mm2        = relation_vector_kernel_fp32_avx<_op>(mm2, _mm256_loadu_ps(src1 + i + 2 * simd_w));
        mm3        = relation_vector_kernel_fp32_avx<_op>(mm3, _mm256_loadu_ps(src1 + i + 3 * simd_w));
        PACK_4UINT32X8_TO_UINT32X32_AVX(mm0, mm1, mm2, mm3, dst + i);
    }
    for (uint64_t i = unroll_body; i < length; i++) {
        dst[i] = relation_scalar_kernel_fp32<_op>(src0[i], src1[i]);
    }
    return ppl::common::RC_SUCCESS;
}

template <relation_op_type_t _op>
ppl::common::RetCode relation_ndarray_binary_op_recursive_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const uint64_t *inc0,
    const uint64_t *inc1,
    const uint64_t *inc_out,
    const uint64_t dim,
    const bool has_paralleled,
    uint8_t *dst)
{
    if (dim == dst_shape->GetDimCount() - 1) { // last dim
        const uint64_t simd_w      = 8;
        const uint64_t unroll_len  = simd_w * 4;
        const uint64_t unroll_body = round(dst_shape->GetDim(dim), unroll_len);

        if (dst_shape->GetDim(dim) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
                __m256 mm0_0, mm0_1, mm0_2, mm0_3;
                if (inc0[dim] == 0) {
                    mm0_0 = _mm256_set1_ps(src0[0]);
                    mm0_1 = mm0_0;
                    mm0_2 = mm0_0;
                    mm0_3 = mm0_0;
                } else {
                    mm0_0 = _mm256_loadu_ps(src0 + i + 0 * simd_w);
                    mm0_1 = _mm256_loadu_ps(src0 + i + 1 * simd_w);
                    mm0_2 = _mm256_loadu_ps(src0 + i + 2 * simd_w);
                    mm0_3 = _mm256_loadu_ps(src0 + i + 3 * simd_w);
                }
                __m256 mm1_0, mm1_1, mm1_2, mm1_3;
                if (inc1[dim] == 0) {
                    mm1_0 = _mm256_set1_ps(src1[0]);
                    mm1_1 = mm1_0;
                    mm1_2 = mm1_0;
                    mm1_3 = mm1_0;
                } else {
                    mm1_0 = _mm256_loadu_ps(src1 + i + 0 * simd_w);
                    mm1_1 = _mm256_loadu_ps(src1 + i + 1 * simd_w);
                    mm1_2 = _mm256_loadu_ps(src1 + i + 2 * simd_w);
                    mm1_3 = _mm256_loadu_ps(src1 + i + 3 * simd_w);
                }
                mm0_0 = relation_vector_kernel_fp32_avx<_op>(mm0_0, mm1_0);
                mm0_1 = relation_vector_kernel_fp32_avx<_op>(mm0_1, mm1_1);
                mm0_2 = relation_vector_kernel_fp32_avx<_op>(mm0_2, mm1_2);
                mm0_3 = relation_vector_kernel_fp32_avx<_op>(mm0_3, mm1_3);
                PACK_4UINT32X8_TO_UINT32X32_AVX(mm0_0, mm0_1, mm0_2, mm0_3, dst + i);
            }
            for (int64_t i = unroll_body; i < dst_shape->GetDim(dim); i++) {
                dst[i] = relation_scalar_kernel_fp32<_op>(src0[i * inc0[dim]], src1[i * inc1[dim]]);
            }
        } else {
            for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
                __m256 mm0_0, mm0_1, mm0_2, mm0_3;
                if (inc0[dim] == 0) {
                    mm0_0 = _mm256_set1_ps(src0[0]);
                    mm0_1 = mm0_0;
                    mm0_2 = mm0_0;
                    mm0_3 = mm0_0;
                } else {
                    mm0_0 = _mm256_loadu_ps(src0 + i + 0 * simd_w);
                    mm0_1 = _mm256_loadu_ps(src0 + i + 1 * simd_w);
                    mm0_2 = _mm256_loadu_ps(src0 + i + 2 * simd_w);
                    mm0_3 = _mm256_loadu_ps(src0 + i + 3 * simd_w);
                }
                __m256 mm1_0, mm1_1, mm1_2, mm1_3;
                if (inc1[dim] == 0) {
                    mm1_0 = _mm256_set1_ps(src1[0]);
                    mm1_1 = mm1_0;
                    mm1_2 = mm1_0;
                    mm1_3 = mm1_0;
                } else {
                    mm1_0 = _mm256_loadu_ps(src1 + i + 0 * simd_w);
                    mm1_1 = _mm256_loadu_ps(src1 + i + 1 * simd_w);
                    mm1_2 = _mm256_loadu_ps(src1 + i + 2 * simd_w);
                    mm1_3 = _mm256_loadu_ps(src1 + i + 3 * simd_w);
                }
                mm0_0 = relation_vector_kernel_fp32_avx<_op>(mm0_0, mm1_0);
                mm0_1 = relation_vector_kernel_fp32_avx<_op>(mm0_1, mm1_1);
                mm0_2 = relation_vector_kernel_fp32_avx<_op>(mm0_2, mm1_2);
                mm0_3 = relation_vector_kernel_fp32_avx<_op>(mm0_3, mm1_3);
                PACK_4UINT32X8_TO_UINT32X32_AVX(mm0_0, mm0_1, mm0_2, mm0_3, dst + i);
            }
            for (int64_t i = unroll_body; i < dst_shape->GetDim(dim); i++) {
                dst[i] = relation_scalar_kernel_fp32<_op>(src0[i * inc0[dim]], src1[i * inc1[dim]]);
            }
        }
    } else {
        if (dst_shape->GetDim(dim) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < dst_shape->GetDim(dim); i++) {
                const float* p_src0 = src0 + i * inc0[dim];
                const float* p_src1 = src1 + i * inc1[dim];
                uint8_t* p_dst      = dst + i * inc_out[dim];
                relation_ndarray_binary_op_recursive_fp32_avx<_op>(
                    src0_shape, src1_shape, dst_shape, p_src0, p_src1, inc0, inc1, inc_out, dim + 1, true, p_dst);
            }
        } else {
            for (int64_t i = 0; i < dst_shape->GetDim(dim); i++) {
                const float* p_src0 = src0 + i * inc0[dim];
                const float* p_src1 = src1 + i * inc1[dim];
                uint8_t* p_dst      = dst + i * inc_out[dim];
                relation_ndarray_binary_op_recursive_fp32_avx<_op>(
                    src0_shape, src1_shape, dst_shape, p_src0, p_src1, inc0, inc1, inc_out, dim + 1, has_paralleled, p_dst);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

inline ppl::nn::TensorShape pad_shape(
    const ppl::nn::TensorShape *shape,
    const int64_t padded_dim_count)
{
    ppl::nn::TensorShape padded_shape(*shape);
    padded_shape.SetDimCount(padded_dim_count);
    if (shape->IsScalar()) {
        for (int64_t i = 0; i < padded_dim_count; i++) {
            padded_shape.SetDim(i, 1);
        }
    } else {
        const int64_t dim_diff = padded_dim_count - shape->GetDimCount();
        for (int64_t i = 0; i < dim_diff; i++) {
            padded_shape.SetDim(i, 1);
        }
        for (int64_t i = dim_diff; i < padded_dim_count; i++) {
            padded_shape.SetDim(i, shape->GetDim(i - dim_diff));
        }
    }
    return padded_shape;
}

template <relation_op_type_t _op>
ppl::common::RetCode relation_ndarray_binary_op_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst)
{
    // pad input dim
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    ppl::nn::TensorShape padded_tensor_shape0 = pad_shape(src0_shape, dim_count);
    ppl::nn::TensorShape padded_tensor_shape1 = pad_shape(src1_shape, dim_count);

    // prepare incs
    uint64_t inc0[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    uint64_t inc1[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    uint64_t inc_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    uint64_t stride0    = 1;
    uint64_t stride1    = 1;
    uint64_t stride_out = 1;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc0[i]    = padded_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        inc1[i]    = padded_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= padded_tensor_shape0.GetDim(i);
        stride1 *= padded_tensor_shape1.GetDim(i);
        stride_out *= dst_shape->GetDim(i);
    }

    return relation_ndarray_binary_op_recursive_fp32_avx<_op>(
        &padded_tensor_shape0, &padded_tensor_shape1, dst_shape, src0, src1, inc0, inc1, inc_out, 0, false, dst);
}

}}}; // namespace ppl::kernel::x86

#undef PACK_4UINT32X8_TO_UINT32X32_AVX

#endif // __ST_PPL_KERNEL_X86_FP32_RELATION_AVX_RELATION_FP32_AVX_COMMON_H_
