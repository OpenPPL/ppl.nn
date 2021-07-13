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

#ifndef __ST_PPL_KERNEL_X86_FP32_ROIALIGN_H_
#define __ST_PPL_KERNEL_X86_FP32_ROIALIGN_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode roialign_ndarray_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *batch_indices_shape,
    const float *input,
    const float *rois,
    const int64_t *batch_indices,
    const int32_t mode,
    const int32_t output_height,
    const int32_t output_width,
    const int32_t sampling_ratio,
    const float spatial_scale,
    float *output);

ppl::common::RetCode roialign_n16cx_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *batch_indices_shape,
    const float *input,
    const float *rois,
    const int64_t *batch_indices,
    const int32_t mode,
    const int32_t output_height,
    const int32_t output_width,
    const int32_t sampling_ratio,
    const float spatial_scale,
    float *output);

ppl::common::RetCode roialign_n16cx_fp32_avx(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *batch_indices_shape,
    const float *input,
    const float *rois,
    const int64_t *batch_indices,
    const int32_t mode,
    const int32_t output_height,
    const int32_t output_width,
    const int32_t sampling_ratio,
    const float spatial_scale,
    float *output);

#ifdef PPLNN_USE_X86_AVX512
ppl::common::RetCode roialign_n16cx_fp32_avx512(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *batch_indices_shape,
    const float *input,
    const float *rois,
    const int64_t *batch_indices,
    const int32_t mode,
    const int32_t output_height,
    const int32_t output_width,
    const int32_t sampling_ratio,
    const float spatial_scale,
    float *output);
#endif

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_ROIALIGN_H_
