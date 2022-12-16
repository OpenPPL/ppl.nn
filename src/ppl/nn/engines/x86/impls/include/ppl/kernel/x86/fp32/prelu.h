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

#ifndef __ST_PPL_KERNEL_X86_FP32_PRELU_H_
#define __ST_PPL_KERNEL_X86_FP32_PRELU_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode prelu_per_channel_ndarray_fp32_avx(
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const float *slope,
    float *dst);

ppl::common::RetCode prelu_per_channel_n16cx_fp32_avx(
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const float *slope,
    float *dst);

ppl::common::RetCode prelu_per_channel_ndarray_fp32_sse(
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const float *slope,
    float *dst);

ppl::common::RetCode prelu_fp32(
    const ppl::common::isa_t isa,
    const ppl::common::TensorShape *src_shape,
    const float *src,
    const float *slope,
    const bool channel_shared,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
