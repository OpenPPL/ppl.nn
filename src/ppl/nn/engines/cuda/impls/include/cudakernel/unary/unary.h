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

#ifndef PPLCUDA_KERNEL_INCLUDE_UNARY_UNARY_H_
#define PPLCUDA_KERNEL_INCLUDE_UNARY_UNARY_H_
#include "ppl/common/tensor_shape.h"
#include "cudakernel/common/common_param.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAUnaryAbsForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryReluForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnarySigmoidForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnarySqrtForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnarySquareForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryTanHForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryFloorForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryCeilForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryErfForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnarySinForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryCosForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

ppl::common::RetCode PPLCUDAUnaryRoundForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam = nullptr);

#endif // PPLCUDA_KERNEL_INCLUDE_UNARY_UNARY_H_
