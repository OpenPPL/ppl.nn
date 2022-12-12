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
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"

#include <cuda_runtime.h>

struct ConvTransposeKernelParam final {
    uint32_t auto_pad;
    int64_t group;
    std::vector<int32_t> dilations;
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> pads;
    std::vector<int32_t> strides;
    std::vector<int32_t> output_padding;
    std::vector<int32_t> output_shape;
};

uint64_t PPLConvTransposeGetFilterBufSizeCudaFp16(
    const ppl::common::TensorShape* weight_shape);

uint64_t PPLConvTransposeGetCompilationBufSizeCuda(
    ppl::common::TensorShape* input_shape,
    ppl::common::TensorShape* output_shape,
    const ConvTransposeKernelParam* param);

uint64_t PPLConvTransposeGetBufSizeCuda(
    ppl::common::TensorShape* input_shape,
    ppl::common::TensorShape* output_shape,
    const ConvTransposeKernelParam* param);

ppl::common::RetCode PPLCUDAConvTransposeCvt(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* in_filter,
    void* temp_buffer,
    void* out_filter,
    const ppl::common::TensorShape* filter_shape,
    const ConvTransposeKernelParam* param);

ppl::common::RetCode PPLCUDAConvTransposeForward(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const CUfunction function,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const void* rev_flt,
    const void* bias,
    ppl::common::TensorShape* output_shape,
    void* output,
    const ConvTransposeKernelParam* param,
    algo_param_t algo_param,
    fuse_param_t &fuse_param,
    void* temp_buffer);

double PPLCUDAConvTransposeSelectKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    //const CUfunction function,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const void* rev_flt,
    const void* bias,
    void* temp_buffer,
    ppl::common::TensorShape* output_shape,
    void* output,
    const ConvTransposeKernelParam* param,
    algo_param_t& algo_param);

#endif // PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
