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

#ifndef PPLCUDA_KERNEL_INCLUDE_NMS_NMS_H_
#define PPLCUDA_KERNEL_INCLUDE_NMS_NMS_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>
#include <float.h>

int64_t PPLNMSGetTempBufferSize(const ppl::nn::TensorShape* scores_shape);

ppl::common::RetCode PPLCUDANMSForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* boxes_shape,
    const void* boxes,
    ppl::nn::TensorShape* scores_shape,
    const void* scores,
    ppl::nn::TensorShape* output_shape,
    int64_t* output,
    void* temp_buffer,
    int64_t temp_buffer_bytes,
    int device_id,
    int center_point_box,
    int max_output_boxes_per_class,
    float iou_threshold,
    float score_threshold = -FLT_MAX);

#endif // PPLCUDA_KERNEL_INCLUDE_NMS_NMS_H_
