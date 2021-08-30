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

#ifndef __ST_PPL_KERNEL_X86_FP32_DEFORM_CONV2D_DEFORM_IM2COL2D_FP32_H_
#define __ST_PPL_KERNEL_X86_FP32_DEFORM_CONV2D_DEFORM_IM2COL2D_FP32_H_

#include <stdint.h>

namespace ppl { namespace kernel { namespace x86 {

void deform_im2col2d_fp32(
    const float* input,
    const float* offset,
    const float* mask,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t channels,
    const int64_t offset_groups,
    const int64_t dst_h,
    const int64_t dst_w,
    const bool use_mask,
    float* columns);

}}}; // namespace ppl::kernel::x86

#endif
