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

#ifndef PPLCUDA_KERNEL_INCLUDE_PAD_PAD_H_
#define PPLCUDA_KERNEL_INCLUDE_PAD_PAD_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/params/onnx/pad_param.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

#define PAD_PARAM_MAX_DIM_SIZE 5
struct PadKernelParam {
    typedef uint32_t pad_mode_t;
    enum { PAD_MODE_CONSTANT = 0,
           PAD_MODE_REFLECT  = 1,
           PAD_MODE_EDGE     = 2 };

    float constant_value = 0.f;
    pad_mode_t mode      = PAD_MODE_CONSTANT;
};

ppl::common::RetCode PPLCUDAPadForwardImp(
    cudaStream_t stream,
    PadKernelParam param,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* pads_shape,
    const int64_t* pads,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif // PPLCUDA_KERNEL_INCLUDE_PAD_PAD_H_