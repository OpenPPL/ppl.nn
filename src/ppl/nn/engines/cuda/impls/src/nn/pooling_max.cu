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

#include "cudakernel/nn/pooling_max.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#include <float.h>

#define HALF_MIN half(-65504)
#define HALF2_MIN half2(-65504, -65504)
#define PPL_CUDA_HALF2_MAX(a, b)                    \
    do {                                             \
        (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; \
        (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; \
    } while (0)
#define PPL_CUDA_MAX(a, b) a = a > b ? a : b

__device__ inline float numerical_min(float a){
    return -FLT_MAX;
}

__device__ inline int8_t numerical_min(int8_t a){
    return -128;
}

__device__ inline half2 numerical_min(half2 a){
    return HALF2_MIN;
}


template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s2_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    half iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy      = oy * 2 + i - pad_height;
            int ix      = ox * 2 + j - pad_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;
        }
    }

    // pooling max & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll TILE_W
        for (int j = 0; j < TILE_W; j++) {
            half val = iregs[i * 2 + 0][j * 2 + 0];
            val      = __hgt(val, iregs[i * 2 + 0][j * 2 + 1]) ? val : iregs[i * 2 + 0][j * 2 + 1];
            val      = __hgt(val, iregs[i * 2 + 0][j * 2 + 2]) ? val : iregs[i * 2 + 0][j * 2 + 2];
            val      = __hgt(val, iregs[i * 2 + 1][j * 2 + 0]) ? val : iregs[i * 2 + 1][j * 2 + 0];
            val      = __hgt(val, iregs[i * 2 + 1][j * 2 + 1]) ? val : iregs[i * 2 + 1][j * 2 + 1];
            val      = __hgt(val, iregs[i * 2 + 1][j * 2 + 2]) ? val : iregs[i * 2 + 1][j * 2 + 2];
            val      = __hgt(val, iregs[i * 2 + 2][j * 2 + 0]) ? val : iregs[i * 2 + 2][j * 2 + 0];
            val      = __hgt(val, iregs[i * 2 + 2][j * 2 + 1]) ? val : iregs[i * 2 + 2][j * 2 + 1];
            val      = __hgt(val, iregs[i * 2 + 2][j * 2 + 2]) ? val : iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s2(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

      if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - pad_height;
            int ix = ox * 2 + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

      // pooling max & store output
#pragma unroll TILE_H
      for (int i = 0; i < TILE_H; i++) {
#pragma unroll TILE_W
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val = (val > iregs[i * 2 + 0][j * 2 + 1]) ? val : iregs[i * 2 + 0][j * 2 + 1];
            val = (val > iregs[i * 2 + 0][j * 2 + 2]) ? val : iregs[i * 2 + 0][j * 2 + 2];
            val = (val > iregs[i * 2 + 1][j * 2 + 0]) ? val : iregs[i * 2 + 1][j * 2 + 0];
            val = (val > iregs[i * 2 + 1][j * 2 + 1]) ? val : iregs[i * 2 + 1][j * 2 + 1];
            val = (val > iregs[i * 2 + 1][j * 2 + 2]) ? val : iregs[i * 2 + 1][j * 2 + 2];
            val = (val > iregs[i * 2 + 2][j * 2 + 0]) ? val : iregs[i * 2 + 2][j * 2 + 0];
            val = (val > iregs[i * 2 + 2][j * 2 + 1]) ? val : iregs[i * 2 + 2][j * 2 + 1];
            val = (val > iregs[i * 2 + 2][j * 2 + 2]) ? val : iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s2(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - pad_height;
            int ix = ox * 2 + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

      // pooling max & store output
#pragma unroll TILE_H
      for (int i = 0; i < TILE_H; i++) {
#pragma unroll TILE_W
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val = (val > iregs[i * 2 + 0][j * 2 + 1]) ? val : iregs[i * 2 + 0][j * 2 + 1];
            val = (val > iregs[i * 2 + 0][j * 2 + 2]) ? val : iregs[i * 2 + 0][j * 2 + 2];
            val = (val > iregs[i * 2 + 1][j * 2 + 0]) ? val : iregs[i * 2 + 1][j * 2 + 0];
            val = (val > iregs[i * 2 + 1][j * 2 + 1]) ? val : iregs[i * 2 + 1][j * 2 + 1];
            val = (val > iregs[i * 2 + 1][j * 2 + 2]) ? val : iregs[i * 2 + 1][j * 2 + 2];
            val = (val > iregs[i * 2 + 2][j * 2 + 0]) ? val : iregs[i * 2 + 2][j * 2 + 0];
            val = (val > iregs[i * 2 + 2][j * 2 + 1]) ? val : iregs[i * 2 + 2][j * 2 + 1];
            val = (val > iregs[i * 2 + 2][j * 2 + 2]) ? val : iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                int res = round((float(val) * in_scale) * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s1_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    half iregs[TILE_H + 2][TILE_W + 2];

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy      = oy + i - pad_height;
            int ix      = ox + j - pad_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;
        }
    }

    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half val = iregs[i + 0][j + 0];
            val      = __hgt(val, iregs[i + 0][j + 1]) ? val : iregs[i + 0][j + 1];
            val      = __hgt(val, iregs[i + 0][j + 2]) ? val : iregs[i + 0][j + 2];
            val      = __hgt(val, iregs[i + 1][j + 0]) ? val : iregs[i + 1][j + 0];
            val      = __hgt(val, iregs[i + 1][j + 1]) ? val : iregs[i + 1][j + 1];
            val      = __hgt(val, iregs[i + 1][j + 2]) ? val : iregs[i + 1][j + 2];
            val      = __hgt(val, iregs[i + 2][j + 0]) ? val : iregs[i + 2][j + 0];
            val      = __hgt(val, iregs[i + 2][j + 1]) ? val : iregs[i + 2][j + 1];
            val      = __hgt(val, iregs[i + 2][j + 2]) ? val : iregs[i + 2][j + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s1(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    T iregs[TILE_H + 2][TILE_W + 2];

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - pad_height;
            int ix = ox + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i + 0][j + 0];
            val = (val > iregs[i + 0][j + 1]) ? val : iregs[i + 0][j + 1];
            val = (val > iregs[i + 0][j + 2]) ? val : iregs[i + 0][j + 2];
            val = (val > iregs[i + 1][j + 0]) ? val : iregs[i + 1][j + 0];
            val = (val > iregs[i + 1][j + 1]) ? val : iregs[i + 1][j + 1];
            val = (val > iregs[i + 1][j + 2]) ? val : iregs[i + 1][j + 2];
            val = (val > iregs[i + 2][j + 0]) ? val : iregs[i + 2][j + 0];
            val = (val > iregs[i + 2][j + 1]) ? val : iregs[i + 2][j + 1];
            val = (val > iregs[i + 2][j + 2]) ? val : iregs[i + 2][j + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s1(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    T iregs[TILE_H + 2][TILE_W + 2];

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - pad_height;
            int ix = ox + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i + 0][j + 0];
            val = (val > iregs[i + 0][j + 1]) ? val : iregs[i + 0][j + 1];
            val = (val > iregs[i + 0][j + 2]) ? val : iregs[i + 0][j + 2];
            val = (val > iregs[i + 1][j + 0]) ? val : iregs[i + 1][j + 0];
            val = (val > iregs[i + 1][j + 1]) ? val : iregs[i + 1][j + 1];
            val = (val > iregs[i + 1][j + 2]) ? val : iregs[i + 1][j + 2];
            val = (val > iregs[i + 2][j + 0]) ? val : iregs[i + 2][j + 0];
            val = (val > iregs[i + 2][j + 1]) ? val : iregs[i + 2][j + 1];
            val = (val > iregs[i + 2][j + 2]) ? val : iregs[i + 2][j + 2];

            if (oy + i < out_height && ox + j < out_width) {
                int res = round((float(val) * in_scale) * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

// #################### pooling max #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half res = HALF_MIN;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy    = (oy + i) * stride_height + fy - pad_height;
                    int ix    = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    half ival = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;

                    res = __hgt(res, ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half(
    const half* input,
    half* output,
    int64_t* indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half res = HALF_MIN;
            int64_t in_index = 0;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy    = (oy + i) * stride_height + fy - pad_height;
                    int ix    = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    half ival = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;

                    if (__hlt(res, ival)) {
                        res = ival;
                        in_index = inOff + iy * in_width + ix;
                    }
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                int64_t out_index = outOff + (oy + i) * out_width + ox + j;
                output[out_index] = res;
                indices[out_index] = in_index;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            T res = numerical_min(T(0));

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                int iy = (oy + i) * stride_height + fy - pad_height;
                int ix = (ox + j) * stride_width + fx - pad_width;
                bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));

                res = (res > ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            int res = numerical_min(T(0));

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                int iy = (oy + i) * stride_height + fy - pad_height;
                int ix = (ox + j) * stride_width + fx - pad_width;
                bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
                res = (res > ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int64_t *indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            T res = numerical_min(T(0));
            int64_t in_index = 0;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy = (oy + i) * stride_height + fy - pad_height;
                    int ix = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
                    if (res < ival) {
                        res = ival;
                        in_index = inOff + iy * in_width + ix;
                    }
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                int64_t out_index = outOff + (oy + i) * out_width + ox + j;
                output[out_index] = res;
                indices[out_index] = in_index;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int64_t *indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            int res = numerical_min(T(0));
            int64_t in_index = 0;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy = (oy + i) * stride_height + fy - pad_height;
                    int ix = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
                    if (res < ival) {
                        res = ival;
                        in_index = inOff + iy * in_width + ix;
                    }
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                int64_t out_index = outOff + (oy + i) * out_width + ox + j;
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[out_index] = res;
                indices[out_index] = in_index;
            }
        }
    }
}

// #################### pooling max f3s2 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s2_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    half2 iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
            iregs[i][j]   = ival;
        }
    }

    // pooling max & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half2 val = iregs[i * 2 + 0][j * 2 + 0];
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 0][j * 2 + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 0][j * 2 + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 1][j * 2 + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 1][j * 2 + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 1][j * 2 + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 2][j * 2 + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 2][j * 2 + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 2][j * 2 + 2]);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

// #################### pooling max f3s1 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s1_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    half2 iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
            iregs[i][j]   = ival;
        }
    }
    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half2 val = iregs[i + 0][j + 0];
            PPL_CUDA_HALF2_MAX(val, iregs[i + 0][j + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 0][j + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 1][j + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 1][j + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 1][j + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 2][j + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 2][j + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 2][j + 2]);

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

// #################### pooling max #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half2 res = HALF2_MIN;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
                    PPL_CUDA_HALF2_MAX(res, ival);
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
#endif
}

template <typename T, int ITER>
__global__ void ppl_cukernel_pooling_max_f2s2_half_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int pad_height,
    int pad_width) // stride is 2
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int ox = hw_idx % out_width;
    int oy = hw_idx / out_width;

    // pooling
    T res;
    half* res_ptr = reinterpret_cast<half*>(&res);
    #pragma unroll
    for (int i = 0; i < ITER; i++) res_ptr[i] = HALF_MIN;
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            // load input
            int ix        = (ox << 1) - pad_width + kx;
            int iy        = (oy << 1) - pad_height + ky;
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
            if (pred) {
                T ival    = input[in_off + in_off_hw];
                half* ival_ptr = reinterpret_cast<half*>(&ival);
                #pragma unroll
                for (int i = 0; i < ITER; ++i) {
                    res_ptr[i] = __hgt(res_ptr[i], ival_ptr[i]) ? res_ptr[i] : ival_ptr[i];
                }
            }
        }
    }
    if (oy < out_height) {
        int out_off_h  = oy * out_width * pad_channels;
        int out_off_w  = ox * pad_channels;
        output[out_off + out_off_h + out_off_w] = res;
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T res = numerical_min(T(0));
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    res = res > ival ? res : ival;
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int64_t* indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T res = numerical_min(T(0));
            int64_t in_index = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    if(res < ival) {
                        res = ival;
                        in_index = in_off + in_off_hw;
                    }
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
                indices[out_off + out_off_h + out_off_w] = in_index;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int res = numerical_min(T(0));
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    res = res > ival ? res : ival;
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int64_t* indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int res = numerical_min(T(0));
            int64_t in_index = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    if(res < ival) {
                        res = ival;
                        in_index = in_off + in_off_hw;
                    }
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = res;
                indices[out_off + out_off_h + out_off_w] = in_index;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= batch * pad_channels * out_width * out_height) return;
    int c_idx = t_idx % pad_channels;
    int hw_idx = t_idx / pad_channels;

    int h_idx = (hw_idx / out_width) % out_height;
    int w_idx = hw_idx % out_width;
    int b_idx = hw_idx / (out_height * out_width);

    int in_h = h_idx * stride_height - padding_height;
    int in_w = w_idx * stride_width - padding_width;

    int in_off = b_idx * in_height * in_width * pad_channels + h_idx * in_width * pad_channels + w_idx * pad_channels + c_idx;
    char4 val = {-128, -128, -128, -128};
    char4 zero = {0, 0, 0, 0};
    char4 *int_input = (char4*)input;
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            int h = in_h + kernel_height;
            int w = in_w + kernel_width;
            bool pred = (w >= 0 && w < in_width) && (h >= 0 && h < in_height);
            char4 src = pred ? int_input[in_off + i * in_width * pad_channels + j * pad_channels] : zero;
            val.x = src.x > val.x ? src.x : val.x;
            val.y = src.y > val.y ? src.y : val.y;
            val.z = src.z > val.z ? src.z : val.z;
            val.w = src.w > val.w ? src.w : val.w;

        }
    }
    output[t_idx] = val;
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s2_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
            iregs[i][j]   = ival;
        }
    }

    // pooling max & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i * 2 + 0][j * 2 + 0];
            PPL_CUDA_MAX(val, iregs[i * 2 + 0][j * 2 + 1]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 0][j * 2 + 2]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 1][j * 2 + 0]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 1][j * 2 + 1]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 1][j * 2 + 2]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 2][j * 2 + 0]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 2][j * 2 + 1]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 2][j * 2 + 2]);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                // int res = round(val * in_scale * out_scale );
                // if(res > 127) res = 127;
                // else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s1_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
            iregs[i][j]   = ival;
        }
    }
    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i + 0][j + 0];
            PPL_CUDA_MAX(val, iregs[i + 0][j + 1]);
            PPL_CUDA_MAX(val, iregs[i + 0][j + 2]);
            PPL_CUDA_MAX(val, iregs[i + 1][j + 0]);
            PPL_CUDA_MAX(val, iregs[i + 1][j + 1]);
            PPL_CUDA_MAX(val, iregs[i + 1][j + 2]);
            PPL_CUDA_MAX(val, iregs[i + 2][j + 0]);
            PPL_CUDA_MAX(val, iregs[i + 2][j + 1]);
            PPL_CUDA_MAX(val, iregs[i + 2][j + 2]);

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                // int res = round(val * in_scale * out_scale );
                // if(res > 127) res = 127;
                // else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const half* input,
    ppl::nn::TensorShape* output_shape,
    half* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool f2 = (kernel_height == 2) && (kernel_width == 2);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            partH      = (out_height + 5) / 6;
            dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_f3s1_half<6, 1><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else {
            ppl_cukernel_pooling_max_common_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int partH             = (out_height + 3) / 4;
        int partW             = (out_width + 0) / 1;
        int padChannelsDivide = (pad_channels >> 1);
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        // dim_grid.y = padChannelsDivide;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_max_f3s1_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else if (f2 && s2) {
            int partH             = out_height;
            int partW             = out_width;
            dim3 dim_block(32, 8, 1);
            dim3 dim_grid;
            dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
            dim_grid.z = batch;
            if (pad_channels >= 256 && (pad_channels % 256 == 0)) {
                int padChannelsDivide = (pad_channels >> 3);
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_f2s2_half_NHWC<float4, 8><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const float4*)input, (float4*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, pad_height, pad_width);
            } else if (pad_channels < 256 && (pad_channels % 64) == 0) {
                int padChannelsDivide = (pad_channels >> 3);
                dim_block.x = 8;
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_f2s2_half_NHWC<float4, 8><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const float4*)input, (float4*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, pad_height, pad_width);
            } else {
                int padChannelsDivide = (pad_channels >> 1);
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_common_half2_NHWC<1, 1><<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
            }
        } else {
            ppl_cukernel_pooling_max_common_half2_NHWC<4, 1><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const half* input,
    ppl::nn::TensorShape* output_shape,
    half* output,
    ppl::nn::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        ppl_cukernel_pooling_max_common_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(
            input, output, indices, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const float* input,
    ppl::nn::TensorShape* output_shape,
    float* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int batch = output_shape->GetDim(0);
    int channels = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height = output_shape->GetDim(2); int out_width = output_shape->GetDim(3);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            partH = (out_height + 5) / 6;
            dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_f3s1<6, 1, float><<<dim_grid, dim_block, 0, stream>>>(
              input, output, batch, pad_channels, in_height, in_width, out_height,
              out_width, kernel_height, kernel_width, stride_height, stride_width,
              pad_height, pad_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2<4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width);
        } else {
            ppl_cukernel_pooling_max_common<4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const float* input,
    ppl::nn::TensorShape* output_shape,
    float* output,
    ppl::nn::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int batch = output_shape->GetDim(0);
    int channels = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height = output_shape->GetDim(2); int out_width = output_shape->GetDim(3);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y; //per thread per chl maxpool
        dim_grid.z = batch;

        ppl_cukernel_pooling_max_common<4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
            input, output, indices, batch, pad_channels, in_height, in_width, out_height,
            out_width, kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width);
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const int8_t* input,
    ppl::nn::TensorShape* output_shape,
    int8_t* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int batch = output_shape->GetDim(0);
    int channels = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height = output_shape->GetDim(2); int out_width = output_shape->GetDim(3);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool f2 = (kernel_height == 2) && (kernel_width == 2);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            partH = (out_height + 5) / 6;
            dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_f3s1<6, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
              input, output, batch, pad_channels, in_height, in_width, out_height,
              out_width, kernel_height, kernel_width, stride_height, stride_width,
              pad_height, pad_width, in_scale, out_scale);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2<4, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_max_common<4, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width, in_scale, out_scale);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int partH             = (out_height + 3) / 4;
        int partW             = (out_width + 0) / 1;
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_max_f3s1_NHWC<4, 1, int8_t><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
            // dim3 dim_block(128, 1, 1);
            // dim3 dim_grid(1,1,1);
            // dim_grid.x = ((pad_channels >> 2) * out_height * out_width * batch + dim_block.x - 1) / dim_block.x;
            // ppl_cukernel_pooling_max_intpacked_NHWC<1,1,char4><<<dim_grid,dim_block,
            //                                                     0,
            //                                                     stream>>>((const char4*)input, (char4*)output, batch, pad_channels >> 2, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2_NHWC<4, 1, int8_t><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        } else if (f2 && s2) {
            dim3 dim_block(128, 1, 1);
            dim3 dim_grid(1,1,1);
            dim_grid.x = ((pad_channels >> 2) * out_height * out_width * batch + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_intpacked_NHWC<1,1,char4><<<dim_grid,dim_block,
                                                                0,
                                                                stream>>>((const char4*)input, (char4*)output, batch, pad_channels >> 2, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_max_common_NHWC<4, 1, int8_t><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const int8_t* input,
    ppl::nn::TensorShape* output_shape,
    int8_t* output,
    ppl::nn::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int batch = output_shape->GetDim(0);
    int channels = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height = output_shape->GetDim(2); int out_width = output_shape->GetDim(3);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y; //per thread per chl maxpool
        dim_grid.z = batch;

        ppl_cukernel_pooling_max_common<4, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
            input, output, indices, batch, pad_channels, in_height, in_width, out_height,
            out_width, kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width, in_scale, out_scale);
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
            int partH             = out_height;
            int partW             = out_width;
            dim3 dim_block(32, 8, 1);
            dim3 dim_grid;
            dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
            dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
            dim_grid.z = batch;
            ppl_cukernel_pooling_max_common_NHWC<1, 1, int8_t><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, indices, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
    return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAMaxPoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAMaxPoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        out_scale = 1.0f / out_scale;
        return PPLCUDAMaxPoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    ppl::nn::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAMaxPoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output,
            indices_shape, indices,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAMaxPoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output,
            indices_shape, indices,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        out_scale = 1.0f / out_scale;
        return PPLCUDAMaxPoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output,
            indices_shape, indices,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
