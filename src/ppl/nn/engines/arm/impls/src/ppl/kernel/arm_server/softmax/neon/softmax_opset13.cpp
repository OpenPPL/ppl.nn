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
#include "ppl/kernel/arm_server/common/math_neon.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

static ppl::common::RetCode softmax_opset13_fp32(
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
    const int64_t dim_count = src_shape->GetDimCount();
    const int64_t axis_dim  = src_shape->GetDim(axis);

    const int64_t simd_w                     = 4;
    const int64_t num_threads                = PPL_OMP_MAX_THREADS();
    const int64_t temp_buffer_len_per_thread = axis_dim * simd_w;
    std::vector<float> temp_base(temp_buffer_len_per_thread * num_threads);

    int64_t outer_dims = 1;
    for (int64_t i = 0; i < axis; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    int64_t inner_dims = 1;
    for (int64_t i = axis + 1; i < dim_count; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t i = 0; i < outer_dims; i++) {
        for (int64_t j_base = 0; j_base < inner_dims; j_base += simd_w) {
            const int64_t thread_id = PPL_OMP_THREAD_ID();
            float *temp             = temp_base.data() + thread_id * temp_buffer_len_per_thread;
            if (inner_dims - j_base >= simd_w) {
                const float *p_src    = src + i * axis_dim * inner_dims + j_base;
                float *p_dst          = dst + i * axis_dim * inner_dims + j_base;
                float32x4_t v_max_val = vdupq_n_f32(numeric_min<float>());

                for (int64_t k = 0; k < axis_dim; k++) {
                    const float32x4_t v_src_val = vld1q_f32(p_src + k * inner_dims);
                    v_max_val                   = vmaxq_f32(v_max_val, v_src_val);
                    vst1q_f32(temp + k * simd_w, v_src_val);
                }

                float sum[4] = {0, 0, 0, 0};
                for (int64_t k = 0; k < axis_dim; k++) {
                    const float exp_val_0 = expf(temp[k * simd_w + 0] - vgetq_lane_f32(v_max_val, 0));
                    const float exp_val_1 = expf(temp[k * simd_w + 1] - vgetq_lane_f32(v_max_val, 1));
                    const float exp_val_2 = expf(temp[k * simd_w + 2] - vgetq_lane_f32(v_max_val, 2));
                    const float exp_val_3 = expf(temp[k * simd_w + 3] - vgetq_lane_f32(v_max_val, 3));
                    temp[k * simd_w + 0]  = exp_val_0;
                    temp[k * simd_w + 1]  = exp_val_1;
                    temp[k * simd_w + 2]  = exp_val_2;
                    temp[k * simd_w + 3]  = exp_val_3;
                    sum[0] += exp_val_0;
                    sum[1] += exp_val_1;
                    sum[2] += exp_val_2;
                    sum[3] += exp_val_3;
                }

                const float32x4_t v_sum   = vld1q_f32(sum);
                const float32x4_t v_r_sum = vdupq_n_f32(1.0f) / v_sum;
                for (int64_t k = 0; k < axis_dim; k++) {
                    const float32x4_t v_dst = vld1q_f32(temp + k * simd_w) * v_r_sum;
                    vst1q_f32(p_dst + k * inner_dims, v_dst);
                }
            } else { // tail process
                for (int64_t j = j_base; j < inner_dims; j++) {
                    const float *p_src = src + i * axis_dim * inner_dims + j;
                    float *p_dst       = dst + i * axis_dim * inner_dims + j;
                    float max_val      = numeric_min<float>();
                    float sum          = 0;

                    for (int64_t k = 0; k < axis_dim; k++) {
                        const float src_val = p_src[k * inner_dims];
                        max_val             = max(max_val, src_val);
                        temp[k]             = src_val;
                    }

                    for (int64_t k = 0; k < axis_dim; k++) {
                        const float src_val = temp[k];
                        float exp_val       = expf(src_val - max_val);
                        temp[k]             = exp_val;
                        sum += exp_val;
                    }

                    const float r_sum = 1.0f / sum;
                    for (int64_t k = 0; k < axis_dim; k++) {
                        p_dst[k * inner_dims] = temp[k] * r_sum;
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
static ppl::common::RetCode softmax_opset13_fp16(
    const ppl::common::TensorShape *src_shape,
    const __fp16 *src,
    const int64_t axis,
    __fp16 *dst)
{
    const int64_t dim_count = src_shape->GetDimCount();
    const int64_t axis_dim  = src_shape->GetDim(axis);

    const int64_t simd_w                     = 4;
    const int64_t num_threads                = PPL_OMP_MAX_THREADS();
    const int64_t temp_buffer_len_per_thread = axis_dim * simd_w;
    std::vector<float> temp_base(temp_buffer_len_per_thread * num_threads);

    int64_t outer_dims = 1;
    for (int64_t i = 0; i < axis; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    int64_t inner_dims = 1;
    for (int64_t i = axis + 1; i < dim_count; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t i = 0; i < outer_dims; i++) {
        for (int64_t j_base = 0; j_base < inner_dims; j_base += simd_w) {
            const int64_t thread_id = PPL_OMP_THREAD_ID();
            float *temp             = temp_base.data() + thread_id * temp_buffer_len_per_thread;

            if (inner_dims - j_base >= simd_w) {
                const __fp16 *p_src   = src + i * axis_dim * inner_dims + j_base;
                __fp16 *p_dst         = dst + i * axis_dim * inner_dims + j_base;
                float32x4_t v_max_val = vdupq_n_f32(numeric_min<float>());

                for (int64_t k = 0; k < axis_dim; k++) {
                    const float32x4_t v_src_val = vcvt_f32_f16(vld1_f16(p_src + k * inner_dims));
                    v_max_val                   = vmaxq_f32(v_max_val, v_src_val);
                    vst1q_f32(temp + k * simd_w, v_src_val);
                }

                float sum[4] = {0, 0, 0, 0};
                for (int64_t k = 0; k < axis_dim; k++) {
                    const float exp_val_0 = expf(temp[k * simd_w + 0] - vgetq_lane_f32(v_max_val, 0));
                    const float exp_val_1 = expf(temp[k * simd_w + 1] - vgetq_lane_f32(v_max_val, 1));
                    const float exp_val_2 = expf(temp[k * simd_w + 2] - vgetq_lane_f32(v_max_val, 2));
                    const float exp_val_3 = expf(temp[k * simd_w + 3] - vgetq_lane_f32(v_max_val, 3));
                    temp[k * simd_w + 0]  = exp_val_0;
                    temp[k * simd_w + 1]  = exp_val_1;
                    temp[k * simd_w + 2]  = exp_val_2;
                    temp[k * simd_w + 3]  = exp_val_3;
                    sum[0] += exp_val_0;
                    sum[1] += exp_val_1;
                    sum[2] += exp_val_2;
                    sum[3] += exp_val_3;
                }

                const float32x4_t v_sum   = vld1q_f32(sum);
                const float32x4_t v_r_sum = vdupq_n_f32(1.0f) / v_sum;
                for (int64_t k = 0; k < axis_dim; k++) {
                    const float32x4_t v_dst = vld1q_f32(temp + k * simd_w) * v_r_sum;
                    vst1_f16(p_dst + k * inner_dims, vcvt_f16_f32(v_dst));
                }
            } else {
                for (int64_t j = j_base; j < inner_dims; j++) {
                    const __fp16 *p_src = src + i * axis_dim * inner_dims + j;
                    __fp16 *p_dst       = dst + i * axis_dim * inner_dims + j;
                    float max_val       = numeric_min<float>();
                    float sum           = 0;

                    for (int64_t k = 0; k < axis_dim; k++) {
                        const float src_val = p_src[k * inner_dims];
                        max_val             = max(max_val, src_val);
                        temp[k]             = src_val;
                    }

                    for (int64_t k = 0; k < axis_dim; k++) {
                        const float src_val = temp[k];
                        float exp_val       = exp(src_val - max_val);
                        temp[k]             = exp_val;
                        sum += exp_val;
                    }

                    const float r_sum = 1.0f / sum;
                    for (int64_t k = 0; k < axis_dim; k++) {
                        p_dst[k * inner_dims] = temp[k] * r_sum;
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}
#endif

ppl::common::RetCode softmax_opset13(
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

    const int64_t fixed_axis = axis < 0 ? axis + src_shape->GetDimCount() : axis;

    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return softmax_opset13_fp32(src_shape, (const float *)src, fixed_axis, (float *)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return softmax_opset13_fp16(src_shape, (const __fp16 *)src, fixed_axis, (__fp16 *)dst);
#endif
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
