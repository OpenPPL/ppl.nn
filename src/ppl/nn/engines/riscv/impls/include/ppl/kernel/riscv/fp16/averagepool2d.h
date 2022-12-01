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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_AVERAFEPOOL2D_H_
#define __ST_PPL_KERNEL_RISCV_FP16_AVERAFEPOOL2D_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode averagepool2d_n8chw_1x16_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const __fp16* src,
    __fp16* dst);

ppl::common::RetCode averagepool2d_nchw_normal_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const __fp16* src,
    __fp16* dst);

ppl::common::RetCode averagepool2d_n8chw_global_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,

    const __fp16* src,
    __fp16* dst);

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_AVERAFEPOOL2D_H_
