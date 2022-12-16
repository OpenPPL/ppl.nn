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

#ifndef __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_ELTWISE_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_ELTWISE_FP32_AVX_H_

#include "arithmetic_kernel_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_eltwise_fp32_avx(
    const ppl::common::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    float *dst)
{
    const int64_t simd_w      = 8;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t length      = dst_shape->CalcElementsIncludingPadding();
    const int64_t unroll_body = round(length, unroll_len);
    const int64_t total_task  = unroll_body / unroll_len;
    std::vector<int64_t> loop_iter(1, max<int64_t>(total_task, 1));
    auto pc = select_single_parallel_loop(
        loop_iter,
        ppl::common::ISA_X86_AVX,
        unroll_len * 2 * sizeof(float),
        unroll_len * sizeof(float),
        unroll_len * sizeof(float),
        1);
    const int64_t task_per_thread = div_up(total_task, pc.num_threads);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t t = 0; t < pc.num_threads; ++t) {
        __m256 zero_vec = _mm256_set1_ps(0.0f);
        const int64_t start_idx = task_per_thread * t * unroll_len;
        const int64_t end_idx = min(task_per_thread * (t + 1), total_task) * unroll_len;
        for (int64_t i = start_idx; i < end_idx; i += unroll_len) {
            __m256 vsrc0_0 = _mm256_loadu_ps(src0 + i + simd_w * 0);
            __m256 vsrc0_1 = _mm256_loadu_ps(src0 + i + simd_w * 1);
            __m256 vsrc0_2 = _mm256_loadu_ps(src0 + i + simd_w * 2);
            __m256 vsrc0_3 = _mm256_loadu_ps(src0 + i + simd_w * 3);

            __m256 vsrc1_0 = _mm256_loadu_ps(src1 + i + simd_w * 0);
            __m256 vsrc1_1 = _mm256_loadu_ps(src1 + i + simd_w * 1);
            __m256 vsrc1_2 = _mm256_loadu_ps(src1 + i + simd_w * 2);
            __m256 vsrc1_3 = _mm256_loadu_ps(src1 + i + simd_w * 3);

            __m256 vdst_0 = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_0, vsrc1_0);
            __m256 vdst_1 = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_1, vsrc1_1);
            __m256 vdst_2 = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_2, vsrc1_2);
            __m256 vdst_3 = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_3, vsrc1_3);

            if (fuse_relu) {
                vdst_0    = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1    = _mm256_max_ps(vdst_1, zero_vec);
                vdst_2    = _mm256_max_ps(vdst_2, zero_vec);
                vdst_3    = _mm256_max_ps(vdst_3, zero_vec);
            }

            _mm256_storeu_ps(dst + i + simd_w * 0, vdst_0);
            _mm256_storeu_ps(dst + i + simd_w * 1, vdst_1);
            _mm256_storeu_ps(dst + i + simd_w * 2, vdst_2);
            _mm256_storeu_ps(dst + i + simd_w * 3, vdst_3);
        }
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = arithmetic_scalar_kernel_fp32_avx<_op>(src0[i], src1[i]);
        if (fuse_relu) {
            dst[i] = max(dst[i], 0.0f);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_ELTWISE_FP32_AVX_H_