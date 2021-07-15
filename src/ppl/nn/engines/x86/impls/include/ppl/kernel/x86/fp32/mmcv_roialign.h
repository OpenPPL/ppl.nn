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

#ifndef __ST_PPL_KERNEL_X86_FP32_MMCV_ROIALIGN_H_
#define __ST_PPL_KERNEL_X86_FP32_MMCV_ROIALIGN_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode mmcv_roialign_ndarray_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

ppl::common::RetCode mmcv_roialign_n16cx_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

ppl::common::RetCode mmcv_roialign_n16cx_fp32_avx(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

#ifdef PPL_USE_X86_AVX512
ppl::common::RetCode mmcv_roialign_n16cx_fp32_avx512(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);
#endif

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_MMCV_ROIALIGN_H_
