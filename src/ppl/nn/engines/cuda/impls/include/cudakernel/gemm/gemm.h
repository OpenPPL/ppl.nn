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
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/nn/conv/conv_fp16.h"

#include "cuda.h"

struct GemmKernelParam final {
    float alpha;
    float beta;
    int32_t transA;
    int32_t transB;
};

uint64_t PPLGemmCUDAGetCompilationBufSize(
    const ppl::common::TensorShape* input_shape,
    conv_param_t& conv_param,
    int transA);

uint64_t PPLGemmCUDAGetRuntimeBufSize(
    const ppl::common::TensorShape* input_shape,
    conv_param_t& conv_param,
    int splitk,
    int splitf,
    int transA);

unsigned int PPLCUDAGemmGetBiasSize(
    const ppl::common::datatype_t infer_type,//output type
    const ppl::common::datatype_t bias_type,
    int K,
    bool is_scalar);

double PPLCUDAGemmJITSelectKernel(
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

double PPLCUDAGemmSelectKernel(
    const cudaDeviceProp& device_prop,
    const cudaStream_t& stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    const GemmKernelParam& param,
    const fuse_param_t& fuse_param,
    algo_param_t& algo_param);

ppl::common::RetCode PPLCUDAGemmForwardImp(
    const cudaDeviceProp& device_prop,
    const cudaStream_t& stream,
    const CUfunction function,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const GemmKernelParam& param,
    void* temp_buffer,
    fuse_param_t& fuse_param,
    const algo_param_t& algo_param);

ppl::common::RetCode PPLCUDAGemmModifyWeights(
    const cudaStream_t& stream,
    ppl::common::TensorShape* weight_shape,
    void* weight,
    void* out_weight,
    const GemmKernelParam* param);

ppl::common::RetCode PPLCUDAGemmModifyBias(
    const cudaStream_t& stream,
    const ppl::common::datatype_t infer_type,
    const ppl::common::TensorShape* bias_shape,
    void* bias,
    const GemmKernelParam* param);

ppl::common::RetCode PPLCUDAGemmModifyWeightsInt8(
    const cudaStream_t &stream,
    ppl::common::TensorShape *weight_shape,
    void *weight,
    void *tmp_weight, // if need transpose
    const GemmKernelParam *param);

double PPLCUDAGemmJITSelectKernelInt8(
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
    quant_param_t &quant_param,
    fuse_param_t& fuse_param,
    algo_param_t& algo_param,
    uint64_t workspace = (uint64_t)8 * 1024 * 1024 * 1024);

double PPLCUDAGemmSelectKernelInt8(
    const cudaDeviceProp& device_prop,
    const cudaStream_t &stream,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *weight_shape,
    const void *weight,
    const void *bias,
    const ppl::common::TensorShape *output_shape,
    void *output,
    void *temp_buffer,
    const GemmKernelParam &param,
    const quant_param_t &quant_param,
    const fuse_param_t &fuse_param,
    algo_param_t &algo_param);

ppl::common::RetCode PPLCUDAGemmForwardImpInt8(
    const cudaDeviceProp& device_prop,
    const cudaStream_t &stream,
    const CUfunction function,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *weight_shape,
    const void *weight,
    const void *bias,
    const ppl::common::TensorShape *output_shape,
    void *output,
    const GemmKernelParam &param,
    void *temp_buffer,
    const quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    const algo_param_t &algo_param);

#endif // PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
