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

#ifndef __ST_PPL_KERNEL_X86_FP32_DEFORM_CONV2D_H_
#define __ST_PPL_KERNEL_X86_FP32_DEFORM_CONV2D_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t deform_conv2d_ref_fp32_get_buffer_bytes(
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t group,
    const int64_t channels,
    const int64_t kernel_h,
    const int64_t kernel_w);

ppl::common::RetCode deform_conv2d_ref_fp32(
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
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
