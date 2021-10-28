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

#ifndef _PPLCUDA_KERNEL_INCLUDE_TOPK_H_
#define _PPLCUDA_KERNEL_INCLUDE_TOPK_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
// #include <cuda_runtime.h>

int64_t PPLTopKGetTempBufferSize(
    const ppl::nn::TensorShape* indices_shape,
    const int K,
    int dim_k,
    bool sorted = true);

ppl::common::RetCode PPLCUDATopKForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* topk_shape,
    void* topk,
    ppl::nn::TensorShape* indices_shape,
    int* indices,
    void* temp_buffer,
    int64_t temp_buffer_bytes,
    int K,
    int dim_k,
    const bool largest = true,
    const bool sorted  = true);

#endif // _PPLCUDA_KERNEL_INCLUDE_TOPK_H_
