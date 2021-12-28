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

#include "cudakernel/nn/global_pooling_max.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#include <float.h>

#define HALF_MIN  half(-65504)
#define HALF2_MIN half2(-65504, -65504)
// #define INT8_MIN int8_t(-128)
#define PPL_CUDA_HALF2_MAX(a, b) \
      do {                                            	\
        (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x;	\
        (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y;	\
      } while (0)
__device__ inline float numerical_min(float a){
    return -FLT_MAX;
}

__device__ inline int8_t numerical_min(int8_t a){
    return -128;
}

__device__ inline half2 numerical_min(half2 a){
    return HALF2_MIN;
}
template<typename T>
__global__ void ppl_cukernel_pooling_max_global_shuffle(
  const T* input,
  T* output,
  int batch,
  int pad_channels,
  int HW)
{
    int c  = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels)
        return;

    T res = numerical_min(T(0));
    for (int i = threadIdx.x * 2; i < HW; i += 64) {
        bool pred0 = i + 0 < HW;
        bool pred1 = i + 1 < HW;
        T ival0 = pred0 ? input[bc * HW + i + 0] : numerical_min(T(0));
        T ival1 = pred1 ? input[bc * HW + i + 1] : numerical_min(T(0));
        res = (res > ival0) ? res : ival0;
        res = (res > ival1) ? res : ival1;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        T sval = __shfl_down_sync(0xffffffff, res, offset);
#else
        T sval = __shfl_down(res, offset);
#endif
        res = (res > sval) ? res : sval;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = res;
}

__global__ void ppl_cukernel_pooling_max_global_shuffle_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c  = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels)
        return;

    half res = HALF_MIN;
    for (int i = threadIdx.x * 2; i < HW; i += 64) {
        bool pred0 = i + 0 < HW;
        bool pred1 = i + 1 < HW;
        half ival0 = pred0 ? input[bc * HW + i + 0] : HALF_MIN;
        half ival1 = pred1 ? input[bc * HW + i + 1] : HALF_MIN;
        res        = __hgt(res, ival0) ? res : ival0;
        res        = __hgt(res, ival1) ? res : ival1;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        half sval = __shfl_down_sync(0xffffffff, res, offset);
#else
        half sval = __shfl_down(res, offset);
#endif
        res = __hgt(res, sval) ? res : sval;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = res;
#endif
}

__global__ void ppl_cukernel_pooling_max_global_shuffle_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    half2 res = HALF2_MIN;
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        PPL_CUDA_HALF2_MAX(res, ival);
    }
    __shared__ half2 sum_buffer[8][32];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            half2 res = sum_buffer[threadIdx.y + i][threadIdx.x];
            PPL_CUDA_HALF2_MAX(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }

    // store output
    if (threadIdx.y == 0)
        output[b_offset + c] = sum_buffer[threadIdx.y][threadIdx.x];
#endif
}

__global__ void ppl_cukernel_pooling_max_global_shuffle_int8_NHWC(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    int8_t res = INT8_MIN;
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        int8_t ival = input[b_offset * HW + i * pad_channels + c];
        res = res > ival ? res : ival;
    }
    __shared__ int8_t sum_buffer[8][32];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            int8_t res = sum_buffer[threadIdx.y + i][threadIdx.x];
            if(res < sum_buffer[threadIdx.y][threadIdx.x]) {
                res = sum_buffer[threadIdx.y][threadIdx.x];
            }
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }

    // store output
    if (threadIdx.y == 0)
        output[b_offset + c] = sum_buffer[threadIdx.y][threadIdx.x];
#endif
}

ppl::common::RetCode PPLCUDAGlobalMaxPoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const half* input,
    ppl::nn::TensorShape* output_shape,
    half* output)
{
    int batch        = output_shape->GetDim(0);
    // int channels = output_shape.GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    // int out_height = output_shape.GetDim(2); int out_width = output_shape.GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_max_global_shuffle_half<<<dim_grid, dim_block, 0, stream>>>((const half*)input, (half*)output, batch, pad_channels, in_height * in_width);
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        // use half2, default padded
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int padChannelsDivide = (pad_channels >> 1); // half2
        int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        dim3 dim_grid(channel_blocks, 1, batch);
        ppl_cukernel_pooling_max_global_shuffle_half2_NHWC<<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height * in_width);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalMaxPoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const float* input,
    ppl::nn::TensorShape* output_shape,
    float* output)
{
    int batch        = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_max_global_shuffle<float><<<dim_grid, dim_block,
            0, stream>>>((const float*)input, (float*)output, batch, pad_channels,
            in_height * in_width);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalMaxPoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape, const int8_t* input,
    ppl::nn::TensorShape* output_shape, int8_t* output) {
    
    int batch = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_max_global_shuffle<int8_t><<<dim_grid, dim_block,
            0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
            in_height * in_width);
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int channel_blocks    = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim3 dim_grid(channel_blocks, 1, batch);
        ppl_cukernel_pooling_max_global_shuffle_int8_NHWC<<<dim_grid,
                                                             dim_block,
                                                             0,
                                                             stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height * in_width);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAGlobalMaxPoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAGlobalMaxPoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        return PPLCUDAGlobalMaxPoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}