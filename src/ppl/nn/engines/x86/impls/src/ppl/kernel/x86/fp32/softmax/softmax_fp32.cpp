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

#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/softmax.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode softmax_ndarray_fp32_ref(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
    const int64_t real_axis = axis < 0 ? axis + src_shape->GetDimCount() : axis;
    if (real_axis < 0 || real_axis >= src_shape->GetDimCount()) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int64_t outer_dim      = 1;
    int64_t inner_dim      = 1;
    for (int64_t i = 0; i < real_axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (int64_t i = real_axis; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < outer_dim; i++) {
        const float *p_src = src + i * inner_dim;
        float *p_dst       = dst + i * inner_dim;

        // find max
        float max_val = p_src[0];
        for (int64_t j = 1; j < inner_dim; j++) {
            if (p_src[j] > max_val) {
                max_val = p_src[j];
            }
        }

        float exp_sum = 0.0f;
        for (int64_t j = 0; j < inner_dim; j++) {
            float exp_val = expf(p_src[j] - max_val);
            p_dst[j]      = exp_val;
            exp_sum += exp_val;
        }
        const float r_exp_sum = 1.0f / exp_sum;
        for (int64_t j = 0; j < inner_dim; j++) {
            p_dst[j] *= r_exp_sum;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode softmax13_ndarray_fp32_ref(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
    const int64_t real_axis = axis < 0 ? axis + src_shape->GetDimCount() : axis;
    if (real_axis < 0 || real_axis >= src_shape->GetDimCount()) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int64_t outer_dim      = 1;
    int64_t inner_dim      = 1;
    int64_t axis_dim       = src_shape->GetDim(real_axis);
    for (int64_t i = 0; i < real_axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (int64_t i = real_axis + 1; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t i = 0; i < outer_dim; i++) {
        for (int64_t k = 0; k < inner_dim; ++k) {
            const float *p_src = src + i * axis_dim * inner_dim + k;
            float *p_dst       = dst + i * axis_dim * inner_dim + k;

            // find max
            float max_val = p_src[0];
            for (int64_t j = 1; j < axis_dim; j++) {
                if (p_src[j * inner_dim] > max_val) {
                    max_val = p_src[j * inner_dim];
                }
            }

            float exp_sum = 0.0f;
            for (int64_t j = 0; j < axis_dim; j++) {
                float exp_val = expf(p_src[j * inner_dim] - max_val);
                p_dst[j * inner_dim] = exp_val;
                exp_sum += exp_val;
            }

            const float r_exp_sum = 1.0f / exp_sum;
            for (int64_t j = 0; j < axis_dim; j++) {
                p_dst[j * inner_dim] *= r_exp_sum;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode softmax_ndarray_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return softmax_ndarray_fp32_avx512(src_shape, src, axis, dst);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return softmax_ndarray_fp32_fma(src_shape, src, axis, dst);
    }
    if (isa & ppl::common::ISA_X86_SSE) {
        return softmax_ndarray_fp32_sse(src_shape, src, axis, dst);
    }
    return softmax_ndarray_fp32_ref(src_shape, src, axis, dst);
}

ppl::common::RetCode softmax13_ndarray_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return softmax13_ndarray_fp32_avx512(src_shape, src, axis, dst);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return softmax13_ndarray_fp32_fma(src_shape, src, axis, dst);
    }
    if (isa & ppl::common::ISA_X86_SSE) {
        return softmax13_ndarray_fp32_sse(src_shape, src, axis, dst);
    }
    return softmax13_ndarray_fp32_ref(src_shape, src, axis, dst);
}

}}} // namespace ppl::kernel::x86
