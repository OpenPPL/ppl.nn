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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_RESIZE_H_
#define __ST_PPL_KERNEL_RISCV_FP32_RESIZE_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode reisze2d_ndarray_pytorch_linear_floor_fp32(const ppl::nn::TensorShape* src_shape,
                                                                const ppl::nn::TensorShape* dst_shape, const float* src,
                                                                const float scale_h, const float scale_w, float* dst);

ppl::common::RetCode reisze2d_ndarray_pytorch_cubic_floor_fp32(const ppl::nn::TensorShape* src_shape,
                                                               const ppl::nn::TensorShape* dst_shape, const float* src,
                                                               const float scale_h, const float scale_w,
                                                               const float cubic_coeff_a, float* dst);

ppl::common::RetCode reisze2d_ndarray_asymmetric_nearest_floor_fp32(const ppl::nn::TensorShape* src_shape,
                                                                    const ppl::nn::TensorShape* dst_shape,
                                                                    const float* src, const float scale_h,
                                                                    const float scale_w, float* dst);

ppl::common::RetCode resize2d_nbcx_pytorch_linear_floor_fp32(const ppl::nn::TensorShape* src_shape,
                                                             const ppl::nn::TensorShape* dst_shape, const float* src,
                                                             const float scale_h, const float scale_w, float* dst);

ppl::common::RetCode resize2d_nbcx_asymmetric_nearest_floor_fp32(const ppl::nn::TensorShape* src_shape,
                                                                 const ppl::nn::TensorShape* dst_shape,
                                                                 const float* src, const float scale_h,
                                                                 const float scale_w, float* dst);

}}}; // namespace ppl::kernel::riscv

#endif
