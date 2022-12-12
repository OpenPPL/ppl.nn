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

#ifndef PPLCUDA_KERNEL_INCLUDE_ROIALIGN_ROIALIGN_H_
#define PPLCUDA_KERNEL_INCLUDE_ROIALIGN_ROIALIGN_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>
#include <float.h>

struct RoiAlignKernelParam final {
    enum { AVG = 0, MAX = 1 };

    uint32_t mode;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t sampling_ratio;
    float spatial_scale;
};

ppl::common::RetCode PPLCUDAROIAlignForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* rois_shape,
    const void* rois,
    ppl::common::TensorShape* batch_indices_shape,
    const void* batch_indices,
    ppl::common::TensorShape* output_shape,
    void* output,
    RoiAlignKernelParam param);
#endif // PPLCUDA_KERNEL_INCLUDE_ONNX_ROIALIGN_ROIALIGN_H_
