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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_LEAKY_RELU_NEON_LEAKY_RELU_H_
#define __ST_PPL_KERNEL_ARM_SERVER_LEAKY_RELU_NEON_LEAKY_RELU_H_

#include "ppl/kernel/arm_server/common/general_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode leaky_relu_fp32(
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode leaky_relu_fp16(
    const ppl::common::TensorShape *src_shape,
    const __fp16 *src,
    const float alpha,
    __fp16 *dst);
#endif

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
