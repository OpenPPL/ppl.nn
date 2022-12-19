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

#include <cmath>
#include <limits>
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode softmax_fp32(
    const ppl::common::TensorShape &input_shape,
    const float *input,
    float *output,
    const uint32_t axis)
{
    uint32_t outer_size = 1;
    uint32_t inner_size = 1;

    // Note: axis is converted to [0, rank(input) ) in the kernel interface.
    // const int64_t axis_size = input_shape.GetDim(axis);
    for (uint32_t idx = 0; idx < axis; idx++) {
        outer_size *= input_shape.GetDim(idx);
    }
    const uint32_t num_dims = input_shape.GetDimCount();
    for (uint32_t idx = axis; idx < num_dims; idx++) {
        inner_size *= input_shape.GetDim(idx);
    }
    const uint32_t inner_size_floor4 = inner_size & (~3);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t out_idx = 0; out_idx < outer_size; out_idx++) {
        const float *input_base = input + out_idx * inner_size;
        float *output_base      = output + out_idx * inner_size;

        float32x4_t vmax = vdupq_n_f32(-numeric_max<float>());
        for (int64_t idx = 0; idx < inner_size_floor4; idx += 4) {
            float32x4_t vin = vld1q_f32(input_base + idx);
            vmax            = vmaxq_f32(vmax, vin);
        }
        float maxval = vgetq_lane_f32(vmax, 0);
        for (int64_t idx = inner_size_floor4; idx < inner_size; idx++) {
            maxval = std::max(maxval, input_base[idx]);
        }
        maxval = std::max(maxval, vgetq_lane_f32(vmax, 1));
        maxval = std::max(maxval, vgetq_lane_f32(vmax, 2));
        maxval = std::max(maxval, vgetq_lane_f32(vmax, 3));

        float32x4_t vexpsum = vdupq_n_f32(0.0f);
        for (int64_t idx = 0; idx < inner_size_floor4; idx += 4) {
            for (int64_t lane = 0; lane < 4; lane++) {
                output_base[idx + lane] = expf((float)input_base[idx + lane] - maxval);
            }
            vexpsum = vaddq_f32(vexpsum, vld1q_f32(output_base + idx));
        }
        float expsum = vgetq_lane_f32(vexpsum, 0);
        for (int64_t idx = inner_size_floor4; idx < inner_size; idx++) {
            float expval     = expf((float)input_base[idx] - maxval);
            output_base[idx] = expval;
            expsum += expval;
        }
        expsum += vgetq_lane_f32(vexpsum, 1);
        expsum += vgetq_lane_f32(vexpsum, 2);
        expsum += vgetq_lane_f32(vexpsum, 3);

        float recp_expsum      = (double)1.0 / expsum;
        float32x4_t vrcpexpsum = vdupq_n_f32(recp_expsum);
        for (int64_t idx = 0; idx < inner_size_floor4; idx += 4) {
            float32x4_t vout = vld1q_f32(output_base + idx);
            vout             = vmulq_f32(vout, vrcpexpsum);
            vst1q_f32(output_base + idx, vout);
        }
        for (int64_t idx = inner_size_floor4; idx < inner_size; idx++) {
            output_base[idx] *= recp_expsum;
        }
    }
    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode softmax_fp16(
    const ppl::common::TensorShape &input_shape,
    const __fp16 *input,
    __fp16 *output,
    const uint32_t axis)
{
    int64_t outer_size = 1;
    int64_t inner_size = 1;

    // Note: axis is converted to [0, rank(input) ) in the kernel interface.
    // const int64_t axis_size = input_shape.GetDim(axis);
    for (int64_t idx = 0; idx < axis; idx++) {
        outer_size *= input_shape.GetDim(idx);
    }
    const int64_t num_dims = input_shape.GetDimCount();
    for (int64_t idx = axis; idx < num_dims; idx++) {
        inner_size *= input_shape.GetDim(idx);
    }
    const int64_t inner_size_floor8 = inner_size & (~7);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t out_idx = 0; out_idx < outer_size; out_idx++) {
        const __fp16 *input_base = input + out_idx * inner_size;
        __fp16 *output_base      = output + out_idx * inner_size;

        float16x8_t vmax = vdupq_n_f16(-numeric_max<float>());
        for (int64_t idx = 0; idx < inner_size_floor8; idx += 8) {
            float16x8_t vin = vld1q_f16(input_base + idx);
            vmax            = vmaxq_f16(vmax, vin);
        }
        __fp16 maxval = vgetq_lane_f16(vmax, 0);
        for (int64_t idx = inner_size_floor8; idx < inner_size; idx++) {
            maxval = std::max(maxval, input_base[idx]);
        }
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 1));
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 2));
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 3));
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 4));
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 5));
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 6));
        maxval = std::max(maxval, vgetq_lane_f16(vmax, 7));

        float16x8_t vexpsum = vdupq_n_f16(0.0f);
        for (int64_t idx = 0; idx < inner_size_floor8; idx += 8) {
            for (int64_t lane = 0; lane < 8; lane++) {
                output_base[idx + lane] = expf((float)input_base[idx + lane] - maxval);
            }
            vexpsum = vaddq_f16(vexpsum, vld1q_f16(output_base + idx));
        }
        float expsum = vgetq_lane_f16(vexpsum, 0);
        for (int64_t idx = inner_size_floor8; idx < inner_size; idx++) {
            float expval     = expf((float)input_base[idx] - maxval);
            output_base[idx] = expval;
            expsum += expval;
        }
        expsum += vgetq_lane_f16(vexpsum, 1);
        expsum += vgetq_lane_f16(vexpsum, 2);
        expsum += vgetq_lane_f16(vexpsum, 3);
        expsum += vgetq_lane_f16(vexpsum, 4);
        expsum += vgetq_lane_f16(vexpsum, 5);
        expsum += vgetq_lane_f16(vexpsum, 6);
        expsum += vgetq_lane_f16(vexpsum, 7);

        float recp_expsum      = (double)1.0 / expsum;
        float16x8_t vrcpexpsum = vdupq_n_f16(recp_expsum);
        for (int64_t idx = 0; idx < inner_size_floor8; idx += 8) {
            float16x8_t vout = vld1q_f16(output_base + idx);
            vout             = vmulq_f16(vout, vrcpexpsum);
            vst1q_f16(output_base + idx, vout);
        }
        for (int64_t idx = inner_size_floor8; idx < inner_size; idx++) {
            output_base[idx] *= recp_expsum;
        }
    }
    return ppl::common::RC_SUCCESS;
}
#endif

ppl::common::RetCode softmax(
    const ppl::common::TensorShape *src_shape,
    const void *src,
    void *dst,
    const int64_t axis)
{
    const auto data_type   = src_shape->GetDataType();
    const auto data_format = src_shape->GetDataFormat();
    if (data_format != ppl::common::DATAFORMAT_NDARRAY) {
        return ppl::common::RC_UNSUPPORTED;
    }

    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return softmax_fp32(*src_shape, (const float *)src, (float *)dst, axis);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return softmax_fp16(*src_shape, (const __fp16 *)src, (__fp16 *)dst, axis);
#endif
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
