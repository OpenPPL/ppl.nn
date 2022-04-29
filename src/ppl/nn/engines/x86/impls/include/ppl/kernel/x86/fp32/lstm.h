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

#ifndef __ST_PPL_KERNEL_X86_FP32_LSTM_H_
#define __ST_PPL_KERNEL_X86_FP32_LSTM_H_

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/kernel/x86/common/rnn_common.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t lstm_fp32_ref_get_buffer_bytes(
    const ppl::nn::TensorShape *X_shape,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c);

ppl::common::RetCode lstm_fp32_ref(
    const ppl::nn::TensorShape *X_shape,
    const float *X,
    const float *X_weight,
    const float *R_weight,
    const float *P_weight,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const float *initial_c,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    void *temp_buffer,
    float *Y,
    float *Y_h,
    float *Y_c);

uint64_t lstm_fp32_fma_get_buffer_bytes(
    const ppl::nn::TensorShape *X_shape,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c);

ppl::common::RetCode lstm_fp32_fma(
    const ppl::nn::TensorShape *X_shape,
    const float *X,
    const float *X_weight,
    const float *R_weight,
    const float *P_weight,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const float *initial_c,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    void *temp_buffer,
    float *Y,
    float *Y_h,
    float *Y_c);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_GEMM_H_
