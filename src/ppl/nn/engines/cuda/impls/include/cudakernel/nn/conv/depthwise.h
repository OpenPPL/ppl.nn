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

#ifndef __PPLCUDA_DEPTHWISE_CONV_H_
#define __PPLCUDA_DEPTHWISE_CONV_H_
#include <cuda_runtime.h>
#include "conv_fp16.h"

int PPLCUDADepthwiseSelectKernel(
    cudaStream_t& stream,
    void* input,
    void* filter,
    void* bias,
    int times,
	conv_param_t &conv_param, 
	fuse_param_t &fuse_param,
    void* output,
    ppl::common::datatype_t type,
    float pic_scale,
    float* flt_scale,
    float out_scale);

void PPLCUDADepthwiseForwardCudaImp(
    cudaStream_t& stream,
    int kernel_id,
    void* input,
    void* filter,
    void* bias,
    conv_param_t &conv_param, 
    fuse_param_t &fuse_param,
    void* output,
    ppl::common::datatype_t type,
    float pic_scale,
    float* flt_scale,
    float out_scale);

void PPLCUDADepthwiseConvertFilter(
    cudaStream_t& stream,
    void* filter,
    void* cvt_filter,
    struct conv_param_t &conv_param,
    ppl::common::datatype_t type);

#endif // __PPLCUDA_DEPTHWISE_CONV_H_
