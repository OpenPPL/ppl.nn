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

#ifndef _X86_KERNEL_LIB_SRC_LIST_X86KERNEL_ARITHMETIC_MULTI_ARRAY_AVX_ARITHMETIC_MULTI_ARRAY_FP32_SSE_H_
#define _X86_KERNEL_LIB_SRC_LIST_X86KERNEL_ARITHMETIC_MULTI_ARRAY_AVX_ARITHMETIC_MULTI_ARRAY_FP32_SSE_H_

#include <nmmintrin.h>
#include <vector>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

enum arithmetic_multi_array_type_t {
    ARRAY_MAX = 0,
    ARRAY_MIN = 1,
    ARRAY_SUM = 2,
};

template <arithmetic_multi_array_type_t _op>
inline float arithmetic_binary_scalar_kernel_fp32(float a, float b);

template <>
inline float arithmetic_binary_scalar_kernel_fp32<ARRAY_MAX>(float a, float b)
{
    return a > b ? a : b;
}
template <>
inline float arithmetic_binary_scalar_kernel_fp32<ARRAY_MIN>(float a, float b)
{
    return a < b ? a : b;
}
template <>
inline float arithmetic_binary_scalar_kernel_fp32<ARRAY_SUM>(float a, float b)
{
    return a + b;
}

template <arithmetic_multi_array_type_t _op> // get input data according to idx & inc_in, have broadcast
inline float arithmetic_multi_scalar_kernel_fp32(
    const float **src_list,
    const uint64_t *inc_in,
    const uint64_t idx,
    const uint64_t num_src)
{
    float result = src_list[0][idx * inc_in[0]];
    for (uint64_t i = 1; i < num_src; i++) {
        result = arithmetic_binary_scalar_kernel_fp32<_op>(result, src_list[i][idx * inc_in[i]]);
    }
    return result;
}

template <arithmetic_multi_array_type_t _op> // get input data according to idx, no broadcast
inline float arithmetic_multi_scalar_kernel_fp32(
    const float **src_list,
    const uint64_t idx,
    const uint64_t num_src)
{
    float result = src_list[0][idx];
    for (uint64_t i = 1; i < num_src; i++) {
        result = arithmetic_binary_scalar_kernel_fp32<_op>(result, src_list[i][idx]);
    }
    return result;
}

template <arithmetic_multi_array_type_t _op>
inline __m128 arithmetic_binary_vector_kernel_fp32_sse(__m128 a, __m128 b);

template <>
inline __m128 arithmetic_binary_vector_kernel_fp32_sse<ARRAY_MAX>(__m128 a, __m128 b)
{
    return _mm_max_ps(a, b);
}
template <>
inline __m128 arithmetic_binary_vector_kernel_fp32_sse<ARRAY_MIN>(__m128 a, __m128 b)
{
    return _mm_min_ps(a, b);
}
template <>
inline __m128 arithmetic_binary_vector_kernel_fp32_sse<ARRAY_SUM>(__m128 a, __m128 b)
{
    return _mm_add_ps(a, b);
}

template <arithmetic_multi_array_type_t _op> // get input data according to idx & inc_in, have broadcast
inline __m128 arithmetic_multi_vector_kernel_fp32_sse(
    const float **src_list,
    const uint64_t *inc_in,
    const uint64_t idx,
    const uint64_t num_src)
{
    __m128 v_result = inc_in[0] == 0 ? _mm_set1_ps(src_list[0][0]) : _mm_loadu_ps(src_list[0] + idx);
    for (uint64_t i = 1; i < num_src; i++) {
        __m128 v_src_list = inc_in[i] == 0 ? _mm_set1_ps(src_list[i][0]) : _mm_loadu_ps(src_list[i] + idx);
        v_result     = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_result, v_src_list);
    }
    return v_result;
}

template <arithmetic_multi_array_type_t _op> // get input data according to idx, no broadcast
inline __m128 arithmetic_multi_vector_kernel_fp32_sse(
    const float **src_list,
    const uint64_t idx,
    const uint64_t num_src)
{
    __m128 v_result = _mm_loadu_ps(src_list[0] + idx);
    for (uint64_t i = 1; i < num_src; i++) {
        v_result = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_result, _mm_loadu_ps(src_list[i] + idx));
    }
    return v_result;
}

template <arithmetic_multi_array_type_t _op, bool _binary>
ppl::common::RetCode arithmetic_multi_array_eltwise_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint64_t num_src,
    float *dst)
{
    const uint64_t simd_w      = 4;
    const uint64_t unroll_len  = simd_w * 4;
    const uint64_t unroll_body = round(dst_shape->GetElementsIncludingPadding(), unroll_len);

    if (_binary) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
            __m128 v_src_list0_0 = _mm_loadu_ps(src_list[0] + i + 0 * simd_w);
            __m128 v_src_list0_1 = _mm_loadu_ps(src_list[0] + i + 1 * simd_w);
            __m128 v_src_list0_2 = _mm_loadu_ps(src_list[0] + i + 2 * simd_w);
            __m128 v_src_list0_3 = _mm_loadu_ps(src_list[0] + i + 3 * simd_w);
            __m128 v_dst_0  = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_0, _mm_loadu_ps(src_list[1] + i + 0 * simd_w));
            __m128 v_dst_1  = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_1, _mm_loadu_ps(src_list[1] + i + 1 * simd_w));
            __m128 v_dst_2  = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_2, _mm_loadu_ps(src_list[1] + i + 2 * simd_w));
            __m128 v_dst_3  = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_3, _mm_loadu_ps(src_list[1] + i + 3 * simd_w));
            _mm_storeu_ps(dst + i + 0 * simd_w, v_dst_0);
            _mm_storeu_ps(dst + i + 1 * simd_w, v_dst_1);
            _mm_storeu_ps(dst + i + 2 * simd_w, v_dst_2);
            _mm_storeu_ps(dst + i + 3 * simd_w, v_dst_3);
        }
        for (uint64_t i = unroll_body; i < dst_shape->GetElementsIncludingPadding(); i++) {
            dst[i] = arithmetic_binary_scalar_kernel_fp32<_op>(src_list[0][i], src_list[1][i]);
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
            __m128 v_dst_0 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, i + 0 * simd_w, num_src);
            __m128 v_dst_1 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, i + 1 * simd_w, num_src);
            __m128 v_dst_2 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, i + 2 * simd_w, num_src);
            __m128 v_dst_3 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, i + 3 * simd_w, num_src);
            _mm_storeu_ps(dst + i + 0 * simd_w, v_dst_0);
            _mm_storeu_ps(dst + i + 1 * simd_w, v_dst_1);
            _mm_storeu_ps(dst + i + 2 * simd_w, v_dst_2);
            _mm_storeu_ps(dst + i + 3 * simd_w, v_dst_3);
        }
        for (uint64_t i = unroll_body; i < dst_shape->GetElementsIncludingPadding(); i++) {
            dst[i] = arithmetic_multi_scalar_kernel_fp32<_op>(src_list, i, num_src);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <arithmetic_multi_array_type_t _op, bool _binary>
void arithmetic_multi_array_ndarray_recursive_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint64_t *inc_in,
    const uint64_t *inc_out,
    const uint64_t num_src,
    const uint64_t dim_idx,
    const bool has_paralleled,
    float *dst)
{
    const uint64_t dim_count = dst_shape->GetDimCount();
    const uint64_t length    = dst_shape->GetDim(dim_idx);

    if (dim_idx == dim_count - 1) { // last dim
        const uint64_t simd_w      = 4;
        const uint64_t unroll_len  = simd_w * 4;
        const uint64_t unroll_body = round(length, unroll_len);

        if (length > 1 && !has_paralleled) {
            if (_binary) {
                const uint64_t inc_in0 = inc_in[0];
                const uint64_t inc_in1 = inc_in[1];
                PRAGMA_OMP_PARALLEL_FOR()
                for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
                    __m128 v_src_list0_0, v_src_list0_1, v_src_list0_2, v_src_list0_3;
                    if (inc_in0 == 0) {
                        v_src_list0_0 = _mm_set1_ps(src_list[0][0]);
                        v_src_list0_1 = v_src_list0_0;
                        v_src_list0_2 = v_src_list0_0;
                        v_src_list0_3 = v_src_list0_0;
                    } else {
                        v_src_list0_0 = _mm_loadu_ps(src_list[0] + i + 0 * simd_w);
                        v_src_list0_1 = _mm_loadu_ps(src_list[0] + i + 1 * simd_w);
                        v_src_list0_2 = _mm_loadu_ps(src_list[0] + i + 2 * simd_w);
                        v_src_list0_3 = _mm_loadu_ps(src_list[0] + i + 3 * simd_w);
                    }

                    __m128 v_src_list1_0, v_src_list1_1, v_src_list1_2, v_src_list1_3;
                    if (inc_in1 == 0) {
                        v_src_list1_0 = _mm_set1_ps(src_list[1][0]);
                        v_src_list1_1 = v_src_list1_0;
                        v_src_list1_2 = v_src_list1_0;
                        v_src_list1_3 = v_src_list1_0;
                    } else {
                        v_src_list1_0 = _mm_loadu_ps(src_list[1] + i + 0 * simd_w);
                        v_src_list1_1 = _mm_loadu_ps(src_list[1] + i + 1 * simd_w);
                        v_src_list1_2 = _mm_loadu_ps(src_list[1] + i + 2 * simd_w);
                        v_src_list1_3 = _mm_loadu_ps(src_list[1] + i + 3 * simd_w);
                    }

                    __m128 v_dst_0 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_0, v_src_list1_0);
                    __m128 v_dst_1 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_1, v_src_list1_1);
                    __m128 v_dst_2 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_2, v_src_list1_2);
                    __m128 v_dst_3 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_3, v_src_list1_3);
                    _mm_storeu_ps(dst + i + 0 * simd_w, v_dst_0);
                    _mm_storeu_ps(dst + i + 1 * simd_w, v_dst_1);
                    _mm_storeu_ps(dst + i + 2 * simd_w, v_dst_2);
                    _mm_storeu_ps(dst + i + 3 * simd_w, v_dst_3);
                }
                for (uint64_t i = unroll_body; i < length; i++) {
                    dst[i] = arithmetic_binary_scalar_kernel_fp32<_op>(src_list[0][i * inc_in0],
                                                                      src_list[1][i * inc_in1]);
                }
            } else {
                PRAGMA_OMP_PARALLEL_FOR()
                for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
                    __m128 v_dst_0 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 0 * simd_w, num_src);
                    __m128 v_dst_1 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 1 * simd_w, num_src);
                    __m128 v_dst_2 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 2 * simd_w, num_src);
                    __m128 v_dst_3 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 3 * simd_w, num_src);
                    _mm_storeu_ps(dst + i + 0 * simd_w, v_dst_0);
                    _mm_storeu_ps(dst + i + 1 * simd_w, v_dst_1);
                    _mm_storeu_ps(dst + i + 2 * simd_w, v_dst_2);
                    _mm_storeu_ps(dst + i + 3 * simd_w, v_dst_3);
                }
                for (uint64_t i = unroll_body; i < length; i++) {
                    dst[i] = arithmetic_multi_scalar_kernel_fp32<_op>(src_list, inc_in, i, num_src);
                }
            }
        } else {
            if (_binary) {
                const uint64_t inc_in0 = inc_in[0];
                const uint64_t inc_in1 = inc_in[1];

                for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
                    __m128 v_src_list0_0, v_src_list0_1, v_src_list0_2, v_src_list0_3;
                    if (inc_in0 == 0) {
                        v_src_list0_0 = _mm_set1_ps(src_list[0][0]);
                        v_src_list0_1 = v_src_list0_0;
                        v_src_list0_2 = v_src_list0_0;
                        v_src_list0_3 = v_src_list0_0;
                    } else {
                        v_src_list0_0 = _mm_loadu_ps(src_list[0] + i + 0 * simd_w);
                        v_src_list0_1 = _mm_loadu_ps(src_list[0] + i + 1 * simd_w);
                        v_src_list0_2 = _mm_loadu_ps(src_list[0] + i + 2 * simd_w);
                        v_src_list0_3 = _mm_loadu_ps(src_list[0] + i + 3 * simd_w);
                    }

                    __m128 v_src_list1_0, v_src_list1_1, v_src_list1_2, v_src_list1_3;
                    if (inc_in1 == 0) {
                        v_src_list1_0 = _mm_set1_ps(src_list[1][0]);
                        v_src_list1_1 = v_src_list1_0;
                        v_src_list1_2 = v_src_list1_0;
                        v_src_list1_3 = v_src_list1_0;
                    } else {
                        v_src_list1_0 = _mm_loadu_ps(src_list[1] + i + 0 * simd_w);
                        v_src_list1_1 = _mm_loadu_ps(src_list[1] + i + 1 * simd_w);
                        v_src_list1_2 = _mm_loadu_ps(src_list[1] + i + 2 * simd_w);
                        v_src_list1_3 = _mm_loadu_ps(src_list[1] + i + 3 * simd_w);
                    }

                    __m128 v_dst_0 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_0, v_src_list1_0);
                    __m128 v_dst_1 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_1, v_src_list1_1);
                    __m128 v_dst_2 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_2, v_src_list1_2);
                    __m128 v_dst_3 = arithmetic_binary_vector_kernel_fp32_sse<_op>(v_src_list0_3, v_src_list1_3);
                    _mm_storeu_ps(dst + i + 0 * simd_w, v_dst_0);
                    _mm_storeu_ps(dst + i + 1 * simd_w, v_dst_1);
                    _mm_storeu_ps(dst + i + 2 * simd_w, v_dst_2);
                    _mm_storeu_ps(dst + i + 3 * simd_w, v_dst_3);
                }
                for (uint64_t i = unroll_body; i < length; i++) {
                    dst[i] = arithmetic_binary_scalar_kernel_fp32<_op>(src_list[0][i * inc_in0],
                                                                      src_list[1][i * inc_in1]);
                }
            } else {
                for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
                    __m128 v_dst_0 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 0 * simd_w, num_src);
                    __m128 v_dst_1 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 1 * simd_w, num_src);
                    __m128 v_dst_2 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 2 * simd_w, num_src);
                    __m128 v_dst_3 = arithmetic_multi_vector_kernel_fp32_sse<_op>(src_list, inc_in, i + 3 * simd_w, num_src);
                    _mm_storeu_ps(dst + i + 0 * simd_w, v_dst_0);
                    _mm_storeu_ps(dst + i + 1 * simd_w, v_dst_1);
                    _mm_storeu_ps(dst + i + 2 * simd_w, v_dst_2);
                    _mm_storeu_ps(dst + i + 3 * simd_w, v_dst_3);
                }
                for (uint64_t i = unroll_body; i < length; i++) {
                    dst[i] = arithmetic_multi_scalar_kernel_fp32<_op>(src_list, inc_in, i, num_src);
                }
            }
        }
    } else {
        if (length > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (uint64_t i = 0; i < length; i++) {
                std::vector<const float*> p_src_list(num_src);
                float *p_dst = dst + i * inc_out[dim_idx];
                for (uint64_t j = 0; j < num_src; j++) {
                    p_src_list[j] = src_list[j] + i * inc_in[j];
                }
                arithmetic_multi_array_ndarray_recursive_fp32_sse<_op, _binary>(
                    dst_shape,
                    p_src_list.data(),
                    inc_in + num_src,
                    inc_out,
                    num_src,
                    dim_idx + 1,
                    true,
                    p_dst);
            }
        } else {
            for (uint64_t i = 0; i < length; i++) {
                std::vector<const float*> p_src_list(num_src);
                float *p_dst = dst + i * inc_out[dim_idx];
                for (uint64_t j = 0; j < num_src; j++) {
                    p_src_list[j] = src_list[j] + i * inc_in[j];
                }
                arithmetic_multi_array_ndarray_recursive_fp32_sse<_op, _binary>(
                    dst_shape,
                    p_src_list.data(),
                    inc_in + num_src,
                    inc_out,
                    num_src,
                    dim_idx + 1,
                    has_paralleled,
                    p_dst);
            }
        }
    }
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

inline uint64_t arithmetic_multi_array_fp32_get_temp_buffer_bytes(const uint64_t num_src)
{
    return (num_src + 1) * PPL_X86_TENSOR_MAX_DIMS() * sizeof(uint64_t);
}

template <arithmetic_multi_array_type_t _op, bool _binary>
ppl::common::RetCode arithmetic_multi_array_ndarray_fp32_sse(
    const ppl::nn::TensorShape **input_shape_list,
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint64_t num_src,
    void *temp_buffer,
    float *dst)
{
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    uint64_t *inc_in         = (uint64_t*)temp_buffer;
    uint64_t *inc_out        = (uint64_t*)temp_buffer + num_src * PPL_X86_TENSOR_MAX_DIMS();

    for (uint64_t i = 0; i < num_src; i++) {
        ppl::nn::TensorShape padded_input_shape = pad_shape(input_shape_list[i], dim_count);
        uint64_t stride                             = 1;
        for (int64_t j = (int64_t)dim_count - 1; j >= 0; j--) {
            inc_in[j * num_src + i] = padded_input_shape.GetDim(j) == 1 ? 0 : stride;
            stride *= padded_input_shape.GetDim(j);
        }
    }
    uint64_t stride_out = 1;
    for (int64_t i = (int64_t)dim_count - 1; i >= 0; i--) {
        inc_out[i] = dst_shape->GetDim(i) == 1 ? 0 : stride_out;
        stride_out *= dst_shape->GetDim(i);
    }

    arithmetic_multi_array_ndarray_recursive_fp32_sse<_op, _binary>(dst_shape, src_list, inc_in, inc_out, num_src, 0, false, dst);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // _X86_KERNEL_LIB_SRC_LIST_X86KERNEL_ARITHMETIC_MULTI_ARRAY_AVX_ARITHMETIC_MULTI_ARRAY_FP32_SSE_H_
