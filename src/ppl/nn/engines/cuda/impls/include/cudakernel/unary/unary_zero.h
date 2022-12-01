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

#ifndef PPLCUDA_KERNEL_INCLUDE_UNARYZERO_UNARYZERO_H_
#define PPLCUDA_KERNEL_INCLUDE_UNARYZERO_UNARYZERO_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/nn/engines/cuda/params/quant_param_cuda.h"
#include "ppl/common/retcode.h"

// these unary ops should specially coded because of f(0) != 0
ppl::common::RetCode PPLCUDAUnaryZeroCosForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryZeroExpForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryZeroLogForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryZeroSigmoidForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);
ppl::common::RetCode PPLCUDAUnaryZeroSoftplusForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);
ppl::common::RetCode PPLCUDAUnaryZeroReciprocalForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);

#endif // PPLCUDA_KERNEL_INCLUDE_UNARYZERO_UNARYZERO_H_
