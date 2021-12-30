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

#include "conv_depthwise_kernel.h"
#include "ppl/nn/common/tensor_shape.h"
#include "cudakernel/nn/conv_fuse_type.h"

#include "depthwise_info.h"

#include <float.h>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define __FLT_MAX__ 3.402823466e+38F
#endif

static std::vector<depthwise_kernel_info> func_vec;

void PPLCUDADepthwiseConvertFilter(
    cudaStream_t& stream,
    void* filter,
    void* cvt_filter,
    struct conv_param_t &conv_param,
    ppl::common::datatype_t type)
{
    int in_height  = conv_param.num_chl;
    int in_width   = conv_param.flt_height * conv_param.flt_width;
    int out_width  = conv_param.num_chl_pad;
    int out_height = in_width;

    int block_size = 32;
    int num_bx     = DivUp(out_height, block_size);
    int num_by     = DivUp(out_width, block_size);
    dim3 dim_grid(num_bx, num_by, 1);
    dim3 dim_block(block_size, 1, 1);

    if(type == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_matrix_transpose<half><<<dim_grid, dim_block, 0, stream>>>(
        (const half*)filter, (half*)cvt_filter, in_height, in_width, out_height, out_width);
    } else if(type == ppl::common::DATATYPE_FLOAT32) {
        ppl_cukernel_matrix_transpose<float><<<dim_grid, dim_block, 0, stream>>>(
        (const float*)filter, (float*)cvt_filter, in_height, in_width, out_height, out_width);
    } else if(type == ppl::common::DATATYPE_INT8) {
        ppl_cukernel_matrix_transpose_int8<<<dim_grid, dim_block, 0, stream>>>(
        (const int8_t*)filter, (int8_t*)cvt_filter, in_height, in_width, out_height, out_width);
    }
    
}
int PPLCUDADepthwiseSelectKernel(
    cudaStream_t& stream,
    void* input,
    void* filter,
    void* bias,
    int times,
	struct conv_param_t &conv_param, 
	struct fuse_param_t &fuse_param,
    void* output,
    ppl::common::datatype_t type,
    float pic_scale,
    float* flt_scale,
    float out_scale)
{
    GETPARAM
    if(func_vec.empty())  InitKernelList(func_vec, type);
    int kernel_id = 0;
    float min_time = FLT_MAX;
    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (uint32_t id = 0; id < func_vec.size(); id++) {
        if (!CanSupport(func_vec[id], conv_param))
            continue;
        cudaEventRecord(begin, stream);
        for (int i = 0; i < 10; i++) {
            int tile_height, tile_width, elems;
            GenConfigure(func_vec[id], conv_param, &tile_height, &tile_width, &elems);
            dim3 dim_block(BLOCK_SIZE,1,1), dim_grid(DivUp(elems,BLOCK_SIZE), 1, 1);
            DivModFast padc_fast(paddingc);
            DivModFast hw_fast(tile_height * tile_width);
            DivModFast width_fast(tile_width);
            if(type == ppl::common::DATATYPE_FLOAT16) {
                func_vec[id].kernel_ptr_half<<<dim_grid, dim_block, 0, stream>>>((const half*)input, (const half*)filter, (const half*)bias, 
                padc_fast, hw_fast, width_fast,
                in_height, in_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w,
                tile_height, tile_width, channels, paddingc, out_height, out_width, 
                in_batch_stride, in_height_stride, in_width_stride, elems, (half*)output, fuse_param);
            } else if(type == ppl::common::DATATYPE_FLOAT32) {
                func_vec[id].kernel_ptr_float<<<dim_grid, dim_block, 0, stream>>>((const float*)input, (const float*)filter, (const float*)bias, 
                padc_fast, hw_fast, width_fast,
                in_height, in_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w,
                tile_height, tile_width, channels, paddingc, out_height, out_width, 
                in_batch_stride, in_height_stride, in_width_stride, elems, (float*)output, fuse_param);
            } else if(type == ppl::common::DATATYPE_INT8) {
                if(func_vec[id].algo_type == SP_DEPTHWISE_KERNEL)
                {
                    dim_block.x = 128;
                    dim_grid.x  = DivUp(DivUp(out_height,4) * out_width * DivUp(channels, 4), 128);
                    dim_grid.y = conv_param.in_num;
                }
                func_vec[id].kernel_ptr_int8<<<dim_grid, dim_block, 0, stream>>>((const int8_t*)input, (const int8_t*)filter, (const float*)bias, 
                padc_fast, hw_fast, width_fast,
                in_height, in_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w,
                tile_height, tile_width, channels, paddingc, out_height, out_width, 
                in_batch_stride, in_height_stride, in_width_stride, elems, (int8_t*)output, fuse_param, pic_scale, flt_scale, out_scale);
            }
        }
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, begin, end);
        if (elapsed < min_time) {
            kernel_id = id;
            min_time  = elapsed;
        }
    }
    return kernel_id;
}
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
    float out_scale)
{

    GETPARAM
    if (func_vec.empty()) InitKernelList(func_vec, type);
    int tile_height, tile_width, elems;
    GenConfigure(func_vec[kernel_id], conv_param, &tile_height, &tile_width, &elems);
    DivModFast padc_fast(paddingc);
    DivModFast hw_fast(tile_height * tile_width);
    DivModFast width_fast(tile_width);
    dim3 dim_block(BLOCK_SIZE,1,1), dim_grid(DivUp(elems, BLOCK_SIZE), 1, 1);
    if(type == ppl::common::DATATYPE_FLOAT16) {
        func_vec[kernel_id].kernel_ptr_half<<<dim_grid, dim_block, 0, stream>>>((const half*)input, (const half*)filter, (const half*)bias, 
        padc_fast, hw_fast, width_fast,
        in_height, in_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w,
        tile_height, tile_width, channels, paddingc, out_height, out_width, 
        in_batch_stride, in_height_stride, in_width_stride, elems, (half*)output, fuse_param);
    } else if(type == ppl::common::DATATYPE_FLOAT32) {
        func_vec[kernel_id].kernel_ptr_float<<<dim_grid, dim_block, 0, stream>>>((const float*)input, (const float*)filter, (const float*)bias, 
        padc_fast, hw_fast, width_fast,
        in_height, in_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w,
        tile_height, tile_width, channels, paddingc, out_height, out_width, 
        in_batch_stride, in_height_stride, in_width_stride, elems, (float*)output, fuse_param);
    } else if(type == ppl::common::DATATYPE_INT8) {
        out_scale = 1.0f / out_scale;
        if(func_vec[kernel_id].algo_type == SP_DEPTHWISE_KERNEL)
        {   
            dim_block.x = 128;
            dim_grid.x  = DivUp(DivUp(out_height,4) * out_width * DivUp(channels, 4), 128);
            dim_grid.y =  conv_param.in_num;
        }
        func_vec[kernel_id].kernel_ptr_int8<<<dim_grid, dim_block, 0, stream>>>((const int8_t*)input, (const int8_t*)filter, (const float*)bias, 
        padc_fast, hw_fast, width_fast,
        in_height, in_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w,
        tile_height, tile_width, channels, paddingc, out_height, out_width, 
        in_batch_stride, in_height_stride, in_width_stride, elems, (int8_t*)output, fuse_param, pic_scale, flt_scale, out_scale);
    }
}


