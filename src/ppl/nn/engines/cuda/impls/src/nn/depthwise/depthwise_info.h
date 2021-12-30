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

#ifndef __PPLCUDA_HALF_DEPTHWISE_INFO_H__
#define __PPLCUDA_HALF_DEPTHWISE_INFO_H__
#include <string>
#include <vector>
#include "conv_depthwise_kernel.h"
#include "sp_depthwise_int8.h"
#include "cudakernel/common/common.h"

#define BLOCK_SIZE 256

typedef void float_depthwise_t(
    const float* input,
    const float* filter,
    const float* bias,
    DivModFast padc_fast,
    DivModFast hw_fast,
    DivModFast width_fast,
    int in_height,
    int in_width,
    int kernel_h,
    int kernel_w,
    int pad_height,
    int pad_width,
    int stride_height,
    int stride_width,
    int hole_h,
    int hole_w,

    int tile_height,
    int tile_width,

    int channels,
    int paddingc,

    int out_height,
    int out_width,

    int in_batch_stride,
    int in_height_stride,
    int in_width_stride,
    int total_elements,
    float* output,
    fuse_param_t fuse_params);

typedef void half_depthwise_t(
    const half* input,
    const half* filter,
    const half* bias,
    DivModFast padc_fast,
    DivModFast hw_fast,
    DivModFast width_fast,
    int in_height,
    int in_width,
    int kernel_h,
    int kernel_w,
    int pad_height,
    int pad_width,
    int stride_height,
    int stride_width,
    int hole_h,
    int hole_w,

    int tile_height,
    int tile_width,

    int channels,
    int paddingc,

    int out_height,
    int out_width,

    int in_batch_stride,
    int in_height_stride,
    int in_width_stride,
    int total_elements,
    half* output,
    fuse_param_t fuse_params);

typedef void int8_depthwise_t(
    const int8_t* input,
    const int8_t* filter,
    const float* bias,
    DivModFast padc_fast,
    DivModFast hw_fast,
    DivModFast width_fast,
    int in_height,
    int in_width,
    int kernel_h,
    int kernel_w,
    int pad_height,
    int pad_width,
    int stride_height,
    int stride_width,
    int hole_h,
    int hole_w,

    int tile_height,
    int tile_width,

    int channels,
    int paddingc,

    int out_height,
    int out_width,

    int in_batch_stride,
    int in_height_stride,
    int in_width_stride,
    int total_elements,
    int8_t* output,
    fuse_param_t fuse_paramse,
    const float pic_scale,
    const float* flt_scale,
    const float out_scale);

enum {
    TILE_DEPTHWISE_KERNEL = 0,
    SP_DEPTHWISE_KERNEL = 1,
};
struct depthwise_kernel_info
{
    /* data */
    half_depthwise_t* kernel_ptr_half;
    float_depthwise_t* kernel_ptr_float;
    int8_depthwise_t* kernel_ptr_int8;

    std::string kernel_name;

    int kernel_index;
    int tile_h;
    int tile_w;

    int in_tile_h;
    int in_tile_w;

    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    int algo_type = 0;


    depthwise_kernel_info(float_depthwise_t *float_kernel, half_depthwise_t *half_kernel, int8_depthwise_t *int8_kernel, 
                    const std::string& name, int kernel_index,
                    int tile_h, int tile_w, int in_tile_h, int in_tile_w,
                    int kernel_h, int kernel_w, int stride_h, int stride_w, int algo_type) {
        kernel_ptr_half = half_kernel;
        kernel_ptr_float = float_kernel;
        kernel_ptr_int8 = int8_kernel;
        this->kernel_name = name;
        this->kernel_index = kernel_index;
        this->tile_h       = tile_h;
        this->tile_w       = tile_w;
        this->in_tile_h    = in_tile_h;     
        this->in_tile_w    = in_tile_w;
        this->kernel_h     = kernel_h;
        this->kernel_w     = kernel_w;
        this->stride_h     = stride_h;
        this->stride_w     = stride_w;
        this->algo_type    = algo_type; 
    }
};

// SRC_TILE_H = (TILE_H - 1) * STRIRDE_H + KERNEL_H

void InitKernelList(std::vector<depthwise_kernel_info> &vec, ppl::common::datatype_t type) {
    int i = vec.size();
    if(type == ppl::common::DATATYPE_FLOAT16) {
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<-1,-1,-1,-1,-1,-1,-1,-1>, nullptr, "ppl_cuda_depthwise_hmma",i,1,1,-1,-1,-1,-1,-1,-1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<2,2,2,2,1,1,1,1>, nullptr, "ppl_cuda_depthwise_hmma",i,2,2,2,2,1,1,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<4,4,4,4,1,1,1,1>, nullptr, "ppl_cuda_depthwise_hmma",i,4,4,4,4,1,1,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<2,2,4,4,3,3,1,1>, nullptr, "ppl_cuda_depthwise_hmma",i,2,2,4,4,3,3,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<4,4,6,6,3,3,1,1>, nullptr, "ppl_cuda_depthwise_hmma",i,4,4,6,6,3,3,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<2,2,5,5,3,3,2,2>, nullptr, "ppl_cuda_depthwise_hmma",i,2,2,5,5,3,3,2,2,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, ppl_cuda_depthwise_hmma<2,2,6,6,5,5,1,1>, nullptr, "ppl_cuda_depthwise_hmma",i,2,2,6,6,5,5,1,1,0)); i++;
    } else if(type == ppl::common::DATATYPE_FLOAT32) {
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<-1,-1,-1,-1,-1,-1,-1,-1>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,1,1,-1,-1,-1,-1,-1,-1,0)); i++;
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<2,2,2,2,1,1,1,1>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,2,2,2,2,1,1,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<4,4,4,4,1,1,1,1>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,4,4,4,4,1,1,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<2,2,4,4,3,3,1,1>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,2,2,4,4,3,3,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<4,4,6,6,3,3,1,1>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,4,4,6,6,3,3,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<2,2,5,5,3,3,2,2>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,2,2,5,5,3,3,2,2,0)); i++;
        vec.push_back(depthwise_kernel_info(ppl_cuda_depthwise_fmma<2,2,6,6,5,5,1,1>, nullptr, nullptr,"ppl_cuda_depthwise_fmma",i,2,2,6,6,5,5,1,1,0)); i++;
    } else if(type == ppl::common::DATATYPE_INT8) {
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<-1,-1,-1,-1,-1,-1,-1,-1>, "ppl_cuda_depthwise_int8mma",i,1,1,-1,-1,-1,-1,-1,-1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<2,2,2,2,1,1,1,1>, "ppl_cuda_depthwise_int8mma",i,2,2,2,2,1,1,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<4,4,4,4,1,1,1,1>, "ppl_cuda_depthwise_int8mma",i,4,4,4,4,1,1,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<2,2,4,4,3,3,1,1>, "ppl_cuda_depthwise_int8mma",i,2,2,4,4,3,3,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<4,4,6,6,3,3,1,1>, "ppl_cuda_depthwise_int8mma",i,4,4,6,6,3,3,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<2,2,5,5,3,3,2,2>, "ppl_cuda_depthwise_int8mma",i,2,2,5,5,3,3,2,2,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8mma<2,2,6,6,5,5,1,1>, "ppl_cuda_depthwise_int8mma",i,2,2,6,6,5,5,1,1,0)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8_nhwc_f3s1<2,2,4,4,3,3,1,1>, "ppl_cuda_depthwise_int8_nhwc_f3s1",i,2,2,4,4,3,3,1,1,1)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8_nhwc_f3s2<2,2,4,4,3,3,2,2>, "ppl_cuda_depthwise_int8_nhwc_f3s2",i,2,2,4,4,3,3,2,2,1)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8_nhwc_f5s1<2,2,4,4,5,5,1,1>, "ppl_cuda_depthwise_int8_nhwc_f5s1",i,2,2,4,4,5,5,1,1,1)); i++;
        vec.push_back(depthwise_kernel_info(nullptr, nullptr, ppl_cuda_depthwise_int8_nhwc_f5s2<2,2,4,4,5,5,2,2>, "ppl_cuda_depthwise_int8_nhwc_f5s1",i,2,2,4,4,5,5,2,2,1)); i++;

    }
}

void GenConfigure(depthwise_kernel_info info, conv_param_t conv_param, int* tile_height, int* tile_width, int* elems)
{
    *tile_height = (conv_param.out_height + info.tile_h - 1) / info.tile_h;
    *tile_width  = (conv_param.out_width + info.tile_w - 1) / info.tile_w;
    *elems       = (*tile_height) * (*tile_width) * conv_param.num_chl_pad * conv_param.in_num;
}

bool CanSupport(depthwise_kernel_info info, conv_param_t conv_param)
{
    return (info.kernel_w == -1 && info.kernel_h == -1) ||
           (conv_param.stride_height == info.stride_h &&
            conv_param.stride_width == info.stride_w && conv_param.flt_height == info.kernel_h &&
            conv_param.flt_width == info.kernel_w &&
            conv_param.hole_height == 1 && conv_param.hole_width == 1);
}

#define GETPARAM                                                                      \
    int in_height        = conv_param.in_height;                                      \
    int in_width         = conv_param.in_width;                                       \
    int kernel_h         = conv_param.flt_height;                                     \
    int kernel_w         = conv_param.flt_width;                                      \
    int pad_h            = conv_param.pad_height;                                     \
    int pad_w            = conv_param.pad_width;                                      \
    int stride_h         = conv_param.stride_height;                                  \
    int stride_w         = conv_param.stride_width;                                   \
    int hole_h           = conv_param.hole_height;                                    \
    int hole_w           = conv_param.hole_width;                                     \
    int out_height       = conv_param.out_height;                                     \
    int out_width        = conv_param.out_width;                                      \
    int channels         = conv_param.num_chl;                                        \
    int paddingc         = conv_param.num_chl_pad;                                    \
    int in_batch_stride  = paddingc * (conv_param.in_height) * (conv_param.in_width); \
    int in_height_stride = conv_param.in_width * paddingc;                            \
    int in_width_stride  = paddingc;
#endif //__PPLCUDA_HALF_DEPTHWISE_INFO_H__
