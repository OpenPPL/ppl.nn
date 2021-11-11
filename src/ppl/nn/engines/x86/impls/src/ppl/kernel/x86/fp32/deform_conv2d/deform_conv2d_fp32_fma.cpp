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

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/fp32/deform_conv2d/deform_im2col2d_fp32.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t deform_conv2d_fp32_fma_get_buffer_bytes(
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t group,
    const int64_t channels,
    const int64_t kernel_h,
    const int64_t kernel_w)
{
    const int64_t ic_per_gp = channels / group;
    if (ic_per_gp * group != channels) {
        return 64u;
    }
    return ic_per_gp * kernel_h * kernel_w * dst_h * dst_w * sizeof(float);
}

ppl::common::RetCode deform_conv2d_fp32_fma(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float *offset,
    const float *mask,
    const float *filter,
    const float *bias,
    const int64_t group,
    const int64_t offset_group,
    const int64_t channels,
    const int64_t num_output,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    void *temp_buffer,
    float *dst)
{
    if (channels % group != 0 || num_output % group != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src_shape->GetDim(0);
    const int64_t src_c = src_shape->GetDim(1);
    const int64_t src_h = src_shape->GetDim(2);
    const int64_t src_w = src_shape->GetDim(3);
    const int64_t dst_c = dst_shape->GetDim(1);
    const int64_t dst_h = dst_shape->GetDim(2);
    const int64_t dst_w = dst_shape->GetDim(3);

    const int64_t ic_per_gp = channels / group;
    const int64_t oc_per_gp = num_output / group;
    float *columns = reinterpret_cast<float*>(temp_buffer);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < group; ++g) {
            deform_im2col2d_fp32(
                src + b * src_c * src_h * src_w + g * ic_per_gp * src_h * src_w,
                offset + b * offset_group * 2 * kernel_h * kernel_w * dst_h * dst_w,
                mask + b * offset_group * kernel_h * kernel_w * dst_h * dst_w,
                src_h, src_w,
                kernel_h, kernel_w,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w,
                ic_per_gp,
                offset_group,
                dst_h, dst_w,
                mask != nullptr,
                columns);
            float *dst_ptr = dst + b * dst_c * dst_h * dst_w + g * oc_per_gp * dst_h * dst_w;
            gemm_fp32_fma(
                filter + g * oc_per_gp * ic_per_gp * kernel_h * kernel_w,
                columns,
                bias + g * oc_per_gp,
                nullptr,
                gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
                bias ? gemm_v_type::COL_VEC : gemm_v_type::EMPTY, gemm_m_type::EMPTY,
                oc_per_gp, dst_h * dst_w, ic_per_gp * kernel_h * kernel_w,
                ic_per_gp * kernel_h * kernel_w, dst_h * dst_w, dst_h * dst_w, dst_h * dst_w,
                1.0f, 1.0f, gemm_post::NONE,
                dst_ptr);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
