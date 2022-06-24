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

#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode hard_swish_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst)
{
    const float alpha = 1.0f / 6.0f;
    const float beta = 0.5f;
    const uint64_t V_REG_ELTS  = 8;
    const uint64_t unroll_len  = V_REG_ELTS * 2;
    const uint64_t unroll_body = round(src_shape->CalcElementsIncludingPadding(), unroll_len);
    if (unroll_body) {
        const auto v_alpha = _mm256_set1_ps(alpha);
        const auto v_beta  = _mm256_set1_ps(beta);
        const auto v_one   = _mm256_set1_ps(1.0f);
        const auto v_zero  = _mm256_setzero_ps();
        PRAGMA_OMP_PARALLEL_FOR()
        for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
            auto v_src0 = _mm256_loadu_ps(src + i + 0 * V_REG_ELTS);
            auto v_src1 = _mm256_loadu_ps(src + i + 1 * V_REG_ELTS);

            auto v_dst0 = _mm256_mul_ps(v_src0, v_alpha);
            auto v_dst1 = _mm256_mul_ps(v_src1, v_alpha);

            v_dst0 = _mm256_add_ps(v_dst0, v_beta);
            v_dst1 = _mm256_add_ps(v_dst1, v_beta);

            v_dst0 = _mm256_min_ps(v_dst0, v_one);
            v_dst1 = _mm256_min_ps(v_dst1, v_one);

            v_dst0 = _mm256_max_ps(v_dst0, v_zero);
            v_dst1 = _mm256_max_ps(v_dst1, v_zero);

            v_dst0 = _mm256_mul_ps(v_dst0, v_src0);
            v_dst1 = _mm256_mul_ps(v_dst1, v_src1);

            _mm256_storeu_ps(dst + i + 0 * V_REG_ELTS, v_dst0);
            _mm256_storeu_ps(dst + i + 1 * V_REG_ELTS, v_dst1);
        }
    }
    for (uint64_t i = unroll_body; i < src_shape->CalcElementsIncludingPadding(); i++) {
        dst[i] = src[i] * max(0.0f, min(1.0f, alpha * src[i] + beta));
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86