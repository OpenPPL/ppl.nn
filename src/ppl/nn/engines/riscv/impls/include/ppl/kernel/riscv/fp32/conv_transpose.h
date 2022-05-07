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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONV_TRANSPOSE_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONV_TRANSPOSE_H_

#include "ppl/kernel/riscv/common/general_include.h"
#include "ppl/nn/engines/riscv/engine_options.h"
#include "ppl/kernel/riscv/common/conv_transpose.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/allocator.h"
#include "ppl/common/sys.h"
#include "functional"

namespace ppl { namespace kernel { namespace riscv {

class conv_transpose_fp32_algo_selector {
public:
    static conv_transpose_common_algo_info select_algo(const ppl::nn::riscv::EngineOptions *engine_options);
};

int64_t conv_transpose_n4cx_get_buffer_bytes_fp32_vec128(
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

ppl::common::RetCode conv_transpose_n4cx_fp32_vec128(
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

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP32_CONV_TRANSPOSE_H_
