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

#ifndef __PPLCUDA_MERGE_SPLIT_H__
#define __PPLCUDA_MERGE_SPLIT_H__

#include <cuda.h>
#include <cuda_fp16.h>
#include "cudakernel/common/macro.h"

//////////////////////////////////////////////////
// merge kernel
//////////////////////////////////////////////////

__global__ void MergeConvSplitResults(
        int4* input,             int4* output, 
	    int split_height_v1,     int split_width_v8, 
	    int out_hw,              int split, 
        int has_bias,            const int4* bias,
        int has_relu,            const __half2 clip_min,
	    bool has_clip,           const __half2 clip_max,
        int has_prelu,           const void* prelu,
        bool has_elt,            const int4* pre_data,
        int has_elt_relu,        const __half2 elt_clip_min,
	    bool has_elt_clip,       const __half2 elt_clip_max,
        int has_elt_prelu,       const void* elt_prelu,
        const __half leaky,      const __half elt_leaky,
        bool has_concat,         int concat_offset_v8,
        int concat_stride_v8);

#endif // __PPLCUDA_MERGE_SPLIT_H__
