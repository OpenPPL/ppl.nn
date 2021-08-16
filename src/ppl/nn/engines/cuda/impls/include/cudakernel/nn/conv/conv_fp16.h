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

#ifndef __PPLCUDA_IMPLICITGEMM_CONV_H_
#define __PPLCUDA_IMPLICITGEMM_CONV_H_
#include <cuda_runtime.h>
#include "ppl/common/types.h"
#include "ppl/common/retcode.h"

struct conv_param_t{
        int in_height;          int in_width;
        int in_num;             int num_grp;
	int num_chl;            int num_chl_pad;
	int num_flt;            int num_flt_pad;
        int flt_height;         int flt_width;
        int out_height;         int out_width;
        int stride_height;      int stride_width;
        int pad_height;         int pad_width;
        int hole_height;        int hole_width;
        int has_bias;           //int4* bias; 
};

struct fuse_param_t{
        int has_activation = 0;// 1: relu,  2: sigmoid
	bool has_clip = false;        float clip_min;     float clip_max; 
        int has_prelu = 0;            void* prelu;      float leaky = 0;       
        bool has_elt  = false;        void* pre_data;    
        int has_elt_activation = 0;     
	bool has_elt_clip = false;    float elt_clip_min; float elt_clip_max; 
        int has_elt_prelu = 0;        void* elt_prelu;  float elt_leaky = 0;    
        bool has_concat   = false;    int concat_offset;       
        int concat_stride;            void* post_concat;
};

struct algo_param_t{
    int kid;
    unsigned int splitk = 1;
    unsigned int splitf = 1;
};
 
int PPLCUDAConvoutionFuseSupport(conv_param_t &conv_param);

uint64_t PPLCUDAConvolutionGetCompilationBufSize(
        ppl::common::datatype_t type, 
        conv_param_t &conv_param,
        uint64_t workspace = ((uint64_t)8)*1024*1024*1024);

uint64_t PPLCUDAConvolutionGetRuntimeBufSize(
        ppl::common::datatype_t type, 
        conv_param_t &conv_param, 
        unsigned int splitk,
        unsigned int splitf,
        uint64_t workspace = ((uint64_t)8)*1024*1024*1024);

ppl::common::RetCode PPLCUDAConvolutionQuickSelectKernel(
        ppl::common::datatype_t type,
        float cash_miss,
        algo_param_t &algo_param,
        conv_param_t &conv_param);

ppl::common::RetCode PPLCUDAConvolutionSelectKernel(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
	int4* bias,
	int4* d_temp_buf, 
        algo_param_t &algo_param,
	conv_param_t &conv_param, 
	fuse_param_t &fuse_param,
	uint64_t workspace = (uint64_t)8*1024*1024*1024);

void PPLCUDAConvolutionForwardImp(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
	int4* bias,
	int4* d_temp_buf, 
        algo_param_t &algo_param,
	conv_param_t &conv_param, 
	fuse_param_t &fuse_param);


#endif// __PPLCUDA_IMPLICITGEMM_CONV_H_
