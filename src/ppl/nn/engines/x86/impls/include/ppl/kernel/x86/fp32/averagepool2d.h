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

#ifndef __ST_PPL_KERNEL_X86_FP32_AVERAGEPOOL2D_H_
#define __ST_PPL_KERNEL_X86_FP32_AVERAGEPOOL2D_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

// averagepool2d n16cx blk

#ifdef PPL_USE_X86_AVX512
ppl::common::RetCode averagepool2d_n16cx_blk1x16_fp32_avx512(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const bool exclusive_mode,
    const bool ceil_mode,
    float *dst);
#endif

ppl::common::RetCode averagepool2d_n16cx_blk1x8_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const bool exclusive_mode,
    const bool ceil_mode,
    float *dst);

ppl::common::RetCode averagepool2d_n16cx_blk1x4_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const bool exclusive_mode,
    const bool ceil_mode,
    float *dst);

// averagepool2d ndarray normal

ppl::common::RetCode averagepool2d_ndarray_normal_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const bool exclusive_mode,
    const bool ceil_mode,
    float *dst);

// maxpool2d n4cx blk
uint64_t averagepool_fp32_get_buffer_bytes(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t pad_w);

ppl::common::RetCode averagepool2d_ndarray_normal_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const bool exclusive_mode,
    const bool ceil_mode,
    void *temp_buffer,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_AVERAGEPOOL_H_
