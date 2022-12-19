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
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode relu_fp32(
    const ppl::common::TensorShape *in_shape,
    const float *input,
    float *output)
{
    int64_t num_elmt               = in_shape->CalcElementsIncludingPadding();
    const int64_t num_elmt_round16 = (num_elmt & (~15));

    if (num_elmt_round16 > 0) {
        PRAGMA_OMP_PARALLEL()
        {
            float32x4_t vzeros = vdupq_n_f32(0.0f);
            PRAGMA_OMP_FOR()
            for (int64_t idx = 0; idx < num_elmt_round16; idx += 16) {
                const float *input_base = input + idx;
                float *output_base      = output + idx;
                float32x4_t vin0        = vld1q_f32(input_base + 0);
                float32x4_t vin1        = vld1q_f32(input_base + 4);
                float32x4_t vin2        = vld1q_f32(input_base + 8);
                float32x4_t vin3        = vld1q_f32(input_base + 12);
                vin0                    = vmaxq_f32(vin0, vzeros);
                vin1                    = vmaxq_f32(vin1, vzeros);
                vin2                    = vmaxq_f32(vin2, vzeros);
                vin3                    = vmaxq_f32(vin3, vzeros);
                vst1q_f32(output_base + 0, vin0);
                vst1q_f32(output_base + 4, vin1);
                vst1q_f32(output_base + 8, vin2);
                vst1q_f32(output_base + 12, vin3);
            }
        }
    }
    for (int64_t idx = num_elmt_round16; idx < num_elmt; idx++) {
        output[idx] = std::max(input[idx], (float)0.0f);
    }

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode relu_fp16(
    const ppl::common::TensorShape *in_shape,
    const __fp16 *input,
    __fp16 *output)
{
    int64_t num_elmt               = in_shape->CalcElementsIncludingPadding();
    const int64_t num_elmt_round32 = (num_elmt & (~31));

    if (num_elmt_round32 > 0) {
        PRAGMA_OMP_PARALLEL()
        {
            float16x8_t vzeros = vdupq_n_f16(0.0f);
            PRAGMA_OMP_FOR()
            for (int64_t idx = 0; idx < num_elmt_round32; idx += 32) {
                const __fp16 *input_base = input + idx;
                __fp16 *output_base      = output + idx;
                float16x8_t vin0         = vld1q_f16(input_base + 0);
                float16x8_t vin1         = vld1q_f16(input_base + 8);
                float16x8_t vin2         = vld1q_f16(input_base + 16);
                float16x8_t vin3         = vld1q_f16(input_base + 24);
                vin0                     = vmaxq_f16(vin0, vzeros);
                vin1                     = vmaxq_f16(vin1, vzeros);
                vin2                     = vmaxq_f16(vin2, vzeros);
                vin3                     = vmaxq_f16(vin3, vzeros);
                vst1q_f16(output_base + 0, vin0);
                vst1q_f16(output_base + 8, vin1);
                vst1q_f16(output_base + 16, vin2);
                vst1q_f16(output_base + 24, vin3);
            }
        }
    }
    for (int64_t idx = num_elmt_round32; idx < num_elmt; idx++) {
        output[idx] = std::max(input[idx], (__fp16)0.0f);
    }

    return ppl::common::RC_SUCCESS;
}
#endif

}}}}; // namespace ppl::kernel::arm_server::neon
