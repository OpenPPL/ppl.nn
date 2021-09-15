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

#include "cudakernel/nn/global_pooling_ave.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>

__global__ void ppl_cukernel_pooling_ave_global_shuffle(
      const float* input,
      float* output,
      int batch,
      int pad_channels,
      int HW)
{
    int c = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels) return;

    float res = float(0);
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        float ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : float(0);
        float ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : float(0);
        float val = ival0 + ival1;
        res = res + val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        float val = __shfl_down_sync(0xffffffff, res, offset);
#else
        float val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = res / HW;
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_half(
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

    half res = half(0);
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        half ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : half(0);
        half ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : half(0);
        half val   = __hadd(ival0, ival1);
        res        = __hadd(res, val);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        half val = __shfl_down_sync(0xffffffff, res, offset);
#else
        half val  = __shfl_down(res, offset);
#endif
        res = __hadd(res, val);
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = __hdiv(res, __float2half(HW));
#endif
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC8(
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

    half2 res = half2(0.f, 0.f);
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        res        = __hadd2(res, ival);
    }
    __shared__ half2 sum_buffer[8][32];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            half2 res                            = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = __hadd2(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0)
        output[b_offset + c] = __h2div(sum_buffer[threadIdx.y][threadIdx.x], half2(HW, HW));
#endif
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpFp16(
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
        ppl_cukernel_pooling_ave_global_shuffle_half<<<dim_grid, dim_block, 0, stream>>>((const half*)input, (half*)output, batch, pad_channels, in_height * in_width);
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        // use half2, default padded
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int padChannelsDivide = (pad_channels >> 1); // half2
        int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        dim3 dim_grid(channel_blocks, 1, batch);
        ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC8<<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height * in_width);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape, const float* input,
    ppl::nn::TensorShape* output_shape, float* output) {
    
    int batch = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_ave_global_shuffle<<<dim_grid, dim_block,
            0, stream>>>((const float*)input, (float*)output, batch, pad_channels,
            in_height * in_width);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAGlobalAvePoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAGlobalAvePoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}