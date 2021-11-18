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

#ifndef PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
#define PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
#include "cudakernel/gemm/gemm.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/convtranspose_param.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"

#include <cuda_runtime.h>


uint64_t pplConvTransposeGetFilterBufSizeCudaFp32(
    const int num_filters,
    const int num_channels,
    const int filter_height,
    const int filter_width);

uint64_t PPLConvTransposeGetBufSizeCuda(
    ppl::nn::TensorShape* input_shape,
    ppl::nn::TensorShape* output_shape,
    const ppl::nn::common::ConvTransposeParam* param);

ppl::common::RetCode PPLCUDAConvTransposeCvt(
    cudaStream_t stream,
    const void* in_filter,
    void* temp_buffer,
    void* out_filter,
    const ppl::nn::TensorShape* filter_shape,
    const ppl::nn::common::ConvTransposeParam* param);

ppl::common::RetCode PPLCUDAConvTransposeForward(
    cudaStream_t stream,
    ppl::nn::cuda::CUDAModule* module,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const void* trans_filter,
    const void* bias,
    const ppl::nn::common::ConvTransposeParam* param,
    algo_param_t algo_param,
    void* temp_buffer,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif // PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
