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

#ifndef PPLCUDA_KERNEL_INCLUDE_SOFTMAX_SOFTMAX_H_
#define PPLCUDA_KERNEL_INCLUDE_SOFTMAX_SOFTMAX_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

uint64_t PPLSoftmaxGetTempBufferSize(const ppl::nn::TensorShape* input_shape, int axis);

ppl::common::RetCode PPLCUDASoftmaxForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis);

template<typename Tin, typename MaskT, typename Tout>
ppl::common::RetCode PPLCUDAFastSoftmaxForwardImp(
    cudaStream_t stream,
    const Tin* input,
    Tout* output,
    const MaskT* key_padding_mask,
    const int mask_type,
    const int B,
    const int H,
    const int T);

ppl::common::RetCode PPLCUDAFastSoftmax(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const void* key_padding_mask,
    const int mask_type);

#endif // PPLCUDA_KERNEL_INCLUDE_SOFTMAX_SOFTMAX_H_
