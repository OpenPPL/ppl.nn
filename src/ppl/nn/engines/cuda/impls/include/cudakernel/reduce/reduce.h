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

#ifndef PPLCUDA_REDUCE_REDUCE_H_
#define PPLCUDA_REDUCE_REDUCE_H_
#include "cudakernel/reduce/reduce_helper.h"
#include "ppl/nn/engines/cuda/params/quant_param_cuda.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/common/tensor_shape.h"

ReduceMode GetReduceMode(PPLReduceDimDes des);

ppl::common::RetCode PPLCUDAReduceForwardImp(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* tmp_buffer = nullptr,
    const ppl::nn::cuda::QuantParamCuda* qparam = nullptr);
#endif
