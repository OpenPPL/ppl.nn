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
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode conv2d_ref_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *sum_src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float *sum_src,
    const float *filter,
    const float *bias,
    const conv2d_fp32_param &param,
    float *dst)
{
    const int64_t batch      = src_shape->GetDim(0);
    const int64_t src_c      = src_shape->GetDim(1);
    const int64_t src_h      = src_shape->GetDim(2);
    const int64_t src_w      = src_shape->GetDim(3);
    const int64_t dst_c      = dst_shape->GetDim(1);
    const int64_t dst_h      = dst_shape->GetDim(2);
    const int64_t dst_w      = dst_shape->GetDim(3);
    const int64_t ic_per_gp  = param.channels / param.group;
    const int64_t oc_per_gp  = param.num_output / param.group;
    const int64_t kernel_h   = param.kernel_h;
    const int64_t kernel_w   = param.kernel_w;
    const int64_t stride_h   = param.stride_h;
    const int64_t stride_w   = param.stride_w;
    const int64_t pad_h      = param.pad_h;
    const int64_t pad_w      = param.pad_w;
    const int64_t dilation_h = param.dilation_h;
    const int64_t dilation_w = param.dilation_w;

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < param.group; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t oc = 0; oc < oc_per_gp; ++oc) {
                for (int64_t oh = 0; oh < dst_h; ++oh) {
                    const float *filter_d = filter + g * oc_per_gp * ic_per_gp * kernel_h * kernel_w;
                    const float *input_d  = src + (b * src_c + g * ic_per_gp) * src_h * src_w;
                    float *output_d       = dst + (b * dst_c + g * oc_per_gp) * dst_h * dst_w;
                    int64_t output_idx    = oc * dst_h * dst_w + oh * dst_w;
                    for (int64_t ow = 0; ow < dst_w; ++ow) {
                        const int64_t ih_start = -pad_h + oh * stride_h;
                        const int64_t iw_start = -pad_w + ow * stride_w;
                        int64_t flt_idx        = oc * ic_per_gp * kernel_h * kernel_w;
                        float sum_val          = 0.0f;
                        for (int64_t ic = 0; ic < ic_per_gp; ++ic) {
                            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                                const int64_t ih   = ih_start + dilation_h * kh;
                                const bool valid_h = (ih >= 0 && ih < src_h);
                                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                    const int64_t iw   = iw_start + dilation_w * kw;
                                    const bool valid_w = (iw >= 0 && iw < src_w);
                                    if (valid_h && valid_w) {
                                        const int64_t input_idx = ic * src_h * src_w + ih * src_w + iw;
                                        sum_val += filter_d[flt_idx] * input_d[input_idx];
                                    }
                                    ++flt_idx;
                                }
                            }
                        }
                        if (bias != nullptr) {
                            sum_val += bias[g * oc_per_gp + oc];
                        }
                        if (param.fuse_flag & conv_fuse_flag::SUM) {
                            const float *sum_d = sum_src + (b * sum_src_shape->GetDim(1) + g * oc_per_gp) * dst_h * dst_w;
                            sum_val += sum_d[output_idx];
                        }
                        if (param.fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
                            sum_val = max(sum_val, 0.0f);
                        }
                        if (param.fuse_flag & conv_fuse_flag::RELU6) {
                            sum_val = min(sum_val, 6.0f);
                        }
                        output_d[output_idx] = sum_val;
                        ++output_idx;
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
