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

#ifndef PPLCUDA_KERNEL_INCLUDE_GEMM_BGEMM_H_
#define PPLCUDA_KERNEL_INCLUDE_GEMM_BGEMM_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/common/retcode.h"
#include "cudakernel/nn/conv/conv_fp16.h"

#include "ppl/nn/engines/cuda/module/cuda_module.h"

#include "cuda.h"

uint64_t PPLBgemmCUDAGetBufSize(
    const ppl::common::TensorShape* input_shape,
    int transA);

unsigned int PPLCUDABgemmGetBiasSize(
    const ppl::common::datatype_t infer_type,//output type
    const ppl::common::datatype_t bias_type,
    int K,
    bool is_scalar);

double PPLCUDABgemmJITSelectKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    ppl::common::datatype_t type,
    ppl::common::TensorShape* input_shape,
    void* input,
    ppl::common::TensorShape* weight_shape,
    void* weight,
    void* bias,
    ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param,
    algo_param_t& algo_param,
    uint64_t workspace = (uint64_t)8 * 1024 * 1024 * 1024);

double PPLCUDABgemmSelectKernel(
    const cudaDeviceProp& device_prop,
    const cudaStream_t& stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    void* weight,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    const ppl::nn::onnx::GemmParam& param,
    const fuse_param_t& fuse_param,
    algo_param_t& algo_param);

ppl::common::RetCode PPLCUDABgemmForwardImp(
    const cudaDeviceProp& device_prop,
    const cudaStream_t& stream,
    const CUfunction function,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    void* weight,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::onnx::GemmParam& param,
    void* temp_buffer,
    fuse_param_t& fuse_param,
    const algo_param_t& algo_param);

ppl::common::RetCode PPLCUDABgemmModifyWeights(
    const cudaStream_t& stream,
    ppl::common::TensorShape* weight_shape,
    void* weight,
    void* out_weight,
    const ppl::nn::onnx::GemmParam* param);

ppl::common::RetCode PPLCUDABgemmCvtOutput(
    const cudaStream_t &stream,
    ppl::common::TensorShape *output_shape,
    void *output,
    void *tmp_output);
ppl::common::RetCode PPLCUDABgemmPadInput(
    const cudaStream_t &stream,
    ppl::common::TensorShape *input_shape,
    void *input,
    void *tmp_input, // if need transpose
    const ppl::nn::onnx::GemmParam *param);


#endif // PPLCUDA_KERNEL_INCLUDE_GEMM_BGEMM_H_
