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

#ifndef PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
#define PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/common/retcode.h"
#include "cudakernel/nn/conv/conv_fp16.h"

#include "ppl/nn/engines/cuda/module/cuda_module.h"

#include "cuda.h"

uint64_t PPLGemmCUDAGetBufSize(
    const ppl::nn::TensorShape* input_shape,
    int transA);

unsigned int PPLCUDAGemmGetBiasSize(
    const ppl::common::datatype_t infer_type,//output type
    const ppl::common::datatype_t bias_type,
    int K,
    bool is_scalar);

double PPLCUDAGemmJITSelectKernel(
    int device_id,
    cudaStream_t& stream,
    ppl::common::datatype_t type,
    ppl::nn::TensorShape* input_shape,
    void* input,
    ppl::nn::TensorShape* weight_shape,
    void* weight,
    void* bias,
    ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param,
    algo_param_t& algo_param,
    uint64_t workspace = (uint64_t)8 * 1024 * 1024 * 1024);

double PPLCUDAGemmSelectKernel(
    const cudaStream_t& stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    const ppl::nn::common::GemmParam& param,
    const fuse_param_t& fuse_param,
    algo_param_t& algo_param);

ppl::common::RetCode PPLCUDAGemmForwardImp(
    const cudaStream_t& stream,
    ppl::nn::cuda::CUDAModule* module,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::common::GemmParam& param,
    void* temp_buffer,
    fuse_param_t& fuse_param,
    const algo_param_t& algo_param);

ppl::common::RetCode PPLCUDAGemmModifyWeights(
    const cudaStream_t& stream,
    ppl::nn::TensorShape* weight_shape,
    void* weight,
    void* out_weight,
    const ppl::nn::common::GemmParam* param);

ppl::common::RetCode PPLCUDAGemmModifyBias(
    const cudaStream_t& stream,
    const ppl::common::datatype_t infer_type,
    const ppl::nn::TensorShape* bias_shape,
    void* bias,
    const ppl::nn::common::GemmParam* param);

ppl::common::RetCode PPLCUDAGemmModifyWeightsInt8(
    const cudaStream_t &stream,
    ppl::nn::TensorShape *weight_shape,
    void *weight,
    void *tmp_weight, // if need transpose
    const ppl::nn::common::GemmParam *param);

double PPLCUDAGemmJITSelectKernelInt8(
    int device_id,
    cudaStream_t& stream,
    ppl::common::datatype_t type,
    ppl::nn::TensorShape* input_shape,
    void* input,
    ppl::nn::TensorShape* weight_shape,
    void* weight,
    void* bias,
    ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    conv_param_t& conv_param,
    quant_param_t &quant_param,
    fuse_param_t& fuse_param,
    algo_param_t& algo_param,
    uint64_t workspace = (uint64_t)8 * 1024 * 1024 * 1024);

double PPLCUDAGemmSelectKernelInt8(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *weight_shape,
    const void *weight,
    const void *bias,
    const ppl::nn::TensorShape *output_shape,
    void *output,
    void *temp_buffer,
    const ppl::nn::common::GemmParam &param,
    const quant_param_t &quant_param,
    const fuse_param_t &fuse_param,
    algo_param_t &algo_param);

ppl::common::RetCode PPLCUDAGemmForwardImpInt8(
    const cudaStream_t &stream,
    ppl::nn::cuda::CUDAModule *module,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *weight_shape,
    const void *weight,
    const void *bias,
    const ppl::nn::TensorShape *output_shape,
    void *output,
    const ppl::nn::common::GemmParam &param,
    void *temp_buffer,
    const quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    const algo_param_t &algo_param);

#endif // PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
