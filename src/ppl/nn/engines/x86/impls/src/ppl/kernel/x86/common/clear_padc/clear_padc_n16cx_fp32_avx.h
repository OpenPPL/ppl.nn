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

#ifndef __ST_PPL_KERNEL_X86_COMMON_CLEAR_PADC_N16CX_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_COMMON_CLEAR_PADC_N16CX_FP32_AVX_H_

#include <immintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

static void clear_padc_n16cx_fp32_avx(const ppl::nn::TensorShape *shape, float *data, const int64_t c_dim_idx = 1)
{
    if (shape->GetDataFormat() == ppl::common::DATAFORMAT_N16CX && shape->GetDim(c_dim_idx) % 16 != 0) { // clear padding channels to 0
        const int64_t simd_w = 8;
        const int64_t c_blk  = 16;

        int64_t outer_dims = 1;
        int64_t inner_dims = 1;
        for (int64_t i = 0; i < c_dim_idx; i++) {
            outer_dims *= shape->GetDim(i);
        }
        for (int64_t i = c_dim_idx + 1; i < shape->GetDimCount(); i++) {
            inner_dims *= shape->GetDim(i);
        }
        const int64_t channels = shape->GetDim(c_dim_idx);
        const int64_t pad_c    = round_up(channels, c_blk);

        uint32_t mask[c_blk] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t c = 0; c < channels - round(channels, c_blk); c++) {
            mask[c] = 0xffffffff;
        }
        const __m256 v_mask_0 = _mm256_loadu_ps((float*)mask + 0 * simd_w);
        const __m256 v_mask_1 = _mm256_loadu_ps((float*)mask + 1 * simd_w);

#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
        for (int64_t od = 0; od < outer_dims; od++) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t id = 0; id < inner_dims; id++) {
                float* p_data   = data + od * pad_c * inner_dims + (pad_c - c_blk) * inner_dims + id * c_blk;
                __m256 v_data_0 = _mm256_loadu_ps(p_data + 0 * simd_w);
                __m256 v_data_1 = _mm256_loadu_ps(p_data + 1 * simd_w);

                v_data_0 = _mm256_and_ps(v_data_0, v_mask_0);
                v_data_1 = _mm256_and_ps(v_data_1, v_mask_1);

                _mm256_storeu_ps(p_data + 0 * simd_w, v_data_0);
                _mm256_storeu_ps(p_data + 1 * simd_w, v_data_1);
            }
        }
    }
}

}}} // namespace ppl::kernel::x86

#endif