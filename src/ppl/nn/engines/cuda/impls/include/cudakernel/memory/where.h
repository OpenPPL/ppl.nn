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

#ifndef PPLCUDA_KERNEL_INCLUDE_WHERE_WHERE_H_
#define PPLCUDA_KERNEL_INCLUDE_WHERE_WHERE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAWhereForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* condition_shape,
    const bool* condition,
    const ppl::nn::TensorShape* input_x_shape,
    const void* input_x,
    const ppl::nn::TensorShape* input_y_shape,
    const void* input_y,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif // PPLCUDA_KERNEL_INCLUDE_WHERE_WHERE_H_
