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

template<typename T>
__global__ void ppl_cukernel_pooling_ave_global_shuffle(
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

    T res = T(0);
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        T ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : T(0);
        T ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : T(0);
        T val = ival0 + ival1;
        res = res + val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        T val = __shfl_down_sync(0xffffffff, res, offset);
#else
        T val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = res / HW;
}


__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8(
      const int8_t* input,
      int8_t* output,
      int batch,
      int pad_channels,
      int HW, float in_scale, float out_scale)
{
    int c = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels) return;

    int32_t res = int32_t(0);
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        int8_t ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : int8_t(0);
        int8_t ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : int8_t(0);
        int32_t val = ival0 + ival1;
        res = res + val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        int32_t val = __shfl_down_sync(0xffffffff, res, offset);
#else
        int32_t val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = round((float(res) / HW * in_scale) / out_scale);
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

    float res = 0.f;
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        half ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : half(0);
        half ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : half(0);
        float val  = __half2float(__hadd(ival0, ival1));
        res        += val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        float val = __shfl_down_sync(0xffffffff, res, offset);
#else
        float val = __shfl_down(res, offset);
#endif
        res +=  val;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = __hdiv(res, __float2half(HW));
#endif
}

static __device__ float2 __f2add(float2 val0, float2 val1) {
    float2 res{0.f, 0.f};
    res.x = val0.x + val1.x;
    res.y = val0.y + val1.y;
    return res;
}

template<int TILE_C, int TILE_HW>
__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC(
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

    float2 res = float2{0.f, 0.f};
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        res        = __f2add(res, __half22float2(ival));
    }
    __shared__ float2 sum_buffer[TILE_HW][TILE_C];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            float2 res                           = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = __f2add(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        float2 res = sum_buffer[threadIdx.y][threadIdx.x];
        res.x = res.x / HW;
        res.y = res.y / HW;
        output[b_offset + c] = __float22half2_rn(res);
    }
#endif
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_atomic(
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

    float2 res = float2{0.f, 0.f};
    // main loop
    for (int i = threadIdx.y + blockDim.y * blockIdx.y;
                i < HW; i += blockDim.y * gridDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        res        = __f2add(res, __half22float2(ival));
    }
    __shared__ float2 sum_buffer[32][8];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            float2 res                           = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = __f2add(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        float2 res = sum_buffer[threadIdx.y][threadIdx.x];
        res.x = res.x / HW;
        res.y = res.y / HW;
        atomicAdd(&output[b_offset + c], __float22half2_rn(res));
    }
#endif
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    int32_t res = 0;
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        int8_t ival = input[b_offset * HW + i * pad_channels + c];
        res = res + ival;
    }
    __shared__ int32_t sum_buffer[8][32];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            int32_t res                            = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                   = res + sum_buffer[threadIdx.y][threadIdx.x];
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0)
        output[b_offset + c] = round((float(sum_buffer[threadIdx.y][threadIdx.x]) / HW * in_scale) / out_scale);
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
    int in_hw = in_height * in_width;

    dim3 dim_grid(1, 1, batch);
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim3 dim_block(32, 4, 1);
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_ave_global_shuffle_half<<<dim_grid, dim_block, 0, stream>>>((const half*)input, (half*)output, batch, pad_channels, in_hw);
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        // use half2, default padded
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int padChannelsDivide = (pad_channels >> 1); // half2
        int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        constexpr int block_threshold = 64;
        constexpr int hw_threshold = 128;
        if (channel_blocks * batch < block_threshold && in_hw > hw_threshold) {
            dim3 dim_block(8, 32, 1); // (c, hw, 1)
            int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
            dim3 dim_grid(channel_blocks, 1, batch);
            if (channel_blocks * batch < block_threshold) {
                int hw_blocks_limit = 8;
                int hw_blocks =  (in_hw + dim_block.y - 1) / dim_block.y;
                hw_blocks = hw_blocks_limit < hw_blocks ? hw_blocks_limit : hw_blocks;
                dim3 dim_grid(channel_blocks, hw_blocks, batch);
                cudaMemsetAsync(output, 0, output_shape->GetBytesIncludingPadding(), stream);
                ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_atomic<<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_hw);
            } else {
                ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC<8, 32><<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_hw);
            }
        } else {
            dim3 dim_grid(channel_blocks, 1, batch);
            ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC<32, 8><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_hw);
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpFp32(
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
        ppl_cukernel_pooling_ave_global_shuffle<float><<<dim_grid, dim_block,
            0, stream>>>((const float*)input, (float*)output, batch, pad_channels,
            in_height * in_width);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape, const int8_t* input,
    ppl::nn::TensorShape* output_shape, int8_t* output, float in_scale, float out_scale) {
    
    int batch = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);
    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_ave_global_shuffle_int8<<<dim_grid, dim_block,
            0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
            in_height * in_width,  in_scale, out_scale);
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        dim3 dim_block(32, 8, 1);
        int channel_blocks    = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim3 dim_grid(channel_blocks, 1, batch);
        ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC<<<dim_grid,
                                                             dim_block,
                                                             0,
                                                             stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height * in_width, in_scale, out_scale);
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
    void* output, float in_scale, float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAGlobalAvePoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAGlobalAvePoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        return PPLCUDAGlobalAvePoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}