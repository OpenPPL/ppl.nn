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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV_TRANSPOSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV_TRANSPOSE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPLNN_USE_X86_AVX512
int64_t conv_transpose_ndarray_fp32_avx512_get_buffer_bytes(
    const int32_t batch,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w);

ppl::common::RetCode conv_transpose_ndarray_fp32_avx512(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    float *tmp_buffer,
    float *output);
#endif

int64_t conv_transpose_ndarray_fp32_fma_get_buffer_bytes(
    const int32_t batch,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w);

ppl::common::RetCode conv_transpose_ndarray_fp32_fma(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    float *tmp_buffer,
    float *output);

int64_t conv_transpose_ndarray_fp32_sse_get_buffer_bytes(
    const int32_t batch,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w);

ppl::common::RetCode conv_transpose_ndarray_fp32_sse(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    float *tmp_buffer,
    float *output);

}}}; // namespace ppl::kernel::x86

#endif
