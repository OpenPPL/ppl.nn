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

#ifndef PPLCUDA_KERNEL_INCLUDE_ONE_HOT_H_
#define PPLCUDA_KERNEL_INCLUDE_ONE_HOT_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/one_hot_param.h"
#include <cuda_runtime.h>

// TODO(@Adam)
ppl::common::RetCode PPLCUDAOneHotForwardImp(
    cudaStream_t stream,
    const void* indices,
    ppl::common::TensorShape* values_shape,
    const void* values,
    ppl::common::TensorShape* output_shape,
    void* output, 
    uint32_t real_axis
  );
#endif // PPLCUDA_KERNEL_INCLUDE_ONE_HOT_H_