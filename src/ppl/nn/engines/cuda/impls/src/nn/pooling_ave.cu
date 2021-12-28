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

#include "cudakernel/nn/pooling_ave.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>


// #################### pooling ave f3s2 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_half(
    const half* input,
    half* output,
    int if_exclude_padding,
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    half iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy      = oy * 2 + i - padding_height;
            int ix      = ox * 2 + j - padding_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : half(0);
        }
    }
    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half val = iregs[i * 2 + 0][j * 2 + 0];
            val      = __hadd(val, iregs[i * 2 + 0][j * 2 + 1]);
            val      = __hadd(val, iregs[i * 2 + 0][j * 2 + 2]);
            val      = __hadd(val, iregs[i * 2 + 1][j * 2 + 0]);
            val      = __hadd(val, iregs[i * 2 + 1][j * 2 + 1]);
            val      = __hadd(val, iregs[i * 2 + 1][j * 2 + 2]);
            val      = __hadd(val, iregs[i * 2 + 2][j * 2 + 0]);
            val      = __hadd(val, iregs[i * 2 + 2][j * 2 + 1]);
            val      = __hadd(val, iregs[i * 2 + 2][j * 2 + 2]);
            if (oy + i < out_height && ox + j < out_width) {
                output[out_off + (oy + i) * out_width + ox + j] = __hdiv(val, cnt);
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s2(
    const T* input, T* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - padding_height;
            int ix = ox * 2 + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : T(0);
        }
    }
    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val = val + iregs[i * 2 + 0][j * 2 + 1];
            val = val + iregs[i * 2 + 0][j * 2 + 2];
            val = val + iregs[i * 2 + 1][j * 2 + 0];
            val = val + iregs[i * 2 + 1][j * 2 + 1];
            val = val + iregs[i * 2 + 1][j * 2 + 2];
            val = val + iregs[i * 2 + 2][j * 2 + 0];
            val = val + iregs[i * 2 + 2][j * 2 + 1];
            val = val + iregs[i * 2 + 2][j * 2 + 2];
            if (oy + i < out_height && ox + j < out_width) {
                output[out_off + (oy + i) * out_width + ox + j] = val / cnt;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_int8(
    const int8_t* input, int8_t* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width, float in_scale, float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    int8_t iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - padding_height;
            int ix = ox * 2 + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : 0;
        }
    }
    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i * 2 + 0][j * 2 + 0];
            val = val + iregs[i * 2 + 0][j * 2 + 1];
            val = val + iregs[i * 2 + 0][j * 2 + 2];
            val = val + iregs[i * 2 + 1][j * 2 + 0];
            val = val + iregs[i * 2 + 1][j * 2 + 1];
            val = val + iregs[i * 2 + 1][j * 2 + 2];
            val = val + iregs[i * 2 + 2][j * 2 + 0];
            val = val + iregs[i * 2 + 2][j * 2 + 1];
            val = val + iregs[i * 2 + 2][j * 2 + 2];
            if (oy + i < out_height && ox + j < out_width) {
                output[out_off + (oy + i) * out_width + ox + j] = round((val / cnt) * in_scale / out_scale);
            }
        }
    }
}

// #################### pooling ave f3s1 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_half(
    const half* input,
    half* output,
    int if_exclude_padding,
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    half iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy      = oy + i - padding_height;
            int ix      = ox + j - padding_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (c < pad_channels);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : half(0);
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half val = iregs[i + 0][j + 0];
            val      = __hadd(val, iregs[i + 0][j + 1]);
            val      = __hadd(val, iregs[i + 0][j + 2]);
            val      = __hadd(val, iregs[i + 1][j + 0]);
            val      = __hadd(val, iregs[i + 1][j + 1]);
            val      = __hadd(val, iregs[i + 1][j + 2]);
            val      = __hadd(val, iregs[i + 2][j + 0]);
            val      = __hadd(val, iregs[i + 2][j + 1]);
            val      = __hadd(val, iregs[i + 2][j + 2]);
            if (oy + i < out_height && ox < out_width) {
                output[out_off + (oy + i) * out_width + ox] = __hdiv(val, cnt);
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s1(
    const T* input, T* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - padding_height;
            int ix = ox + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (c < pad_channels);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : T(0);
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i + 0][j + 0];
            val = val + iregs[i + 0][j + 1];
            val = val + iregs[i + 0][j + 2];
            val = val + iregs[i + 1][j + 0];
            val = val + iregs[i + 1][j + 1];
            val = val + iregs[i + 1][j + 2];
            val = val + iregs[i + 2][j + 0];
            val = val + iregs[i + 2][j + 1];
            val = val + iregs[i + 2][j + 2];
            if (oy + i < out_height && ox < out_width) {
                output[out_off + (oy + i) * out_width + ox] = (val / cnt);
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_int8(
    const int8_t* input, int8_t* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width, float in_scale, float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    int8_t iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - padding_height;
            int ix = ox + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (c < pad_channels);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : 0;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i + 0][j + 0];
            val = val + iregs[i + 0][j + 1];
            val = val + iregs[i + 0][j + 2];
            val = val + iregs[i + 1][j + 0];
            val = val + iregs[i + 1][j + 1];
            val = val + iregs[i + 1][j + 2];
            val = val + iregs[i + 2][j + 0];
            val = val + iregs[i + 2][j + 1];
            val = val + iregs[i + 2][j + 2];
            if (oy + i < out_height && ox < out_width) {
                output[out_off + (oy + i) * out_width + ox] = round((val / cnt) * in_scale / out_scale);
            }
        }
    }
}

// #################### pooling ave #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_half(
    const half* input,
    half* output,
    int if_exclude_padding,
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt  = 0;
            half res = half(0);
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix    = (ox + j) * stride_width - padding_width + kx;
                    int iy    = (oy + i) * stride_height - padding_height + ky;
                    bool pred = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    half ival = pred ? input[in_off + iy * in_width + ix] : half(0);
                    res       = __hadd(res, ival);

                    // cnt exclude padding
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;
            // store output
            res = __hdiv(res, cnt);
            if (ox + j < out_width && oy + i < out_height) {
                output[out_off + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_common(
    const T* input, T* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            T res = T(0);
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix = (ox + j) * stride_width - padding_width + kx;
                    int iy = (oy + i) * stride_height - padding_height + ky;
                    bool pred = (ix >= 0 && ix < in_width) && (iy >=0 && iy < in_height);
                    T ival = pred ? input[in_off + iy * in_width + ix] : T(0);
                    res = res + ival;

                    // cnt exclude padding
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;
            // store output
            res = res / cnt;
            if (ox + j < out_width && oy + i < out_height) {
                output[out_off + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_int8(
    const int8_t* input, int8_t* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width, float in_scale, float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            int32_t res = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix = (ox + j) * stride_width - padding_width + kx;
                    int iy = (oy + i) * stride_height - padding_height + ky;
                    bool pred = (ix >= 0 && ix < in_width) && (iy >=0 && iy < in_height);
                    int8_t ival = pred ? input[in_off + iy * in_width + ix] : 0;
                    res = res + ival;

                    // cnt exclude padding
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;
            // store output
            if (ox + j < out_width && oy + i < out_height) {
                output[out_off + (oy + i) * out_width + ox + j] = round((res / cnt) * in_scale / out_scale);
            }
        }
    }
}

// #################### pooling ave f3s2 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_half2_NHWC(
    const half2* input,
    half2* output,
    int if_exclude_padding,
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
            half2 ival    = pred ? input[in_off + in_off_hw] : half2(0.f, 0.f);
            iregs[i][j]   = ival;
        }
    }

    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half2 val = iregs[i * 2 + 0][j * 2 + 0];
            val       = __hadd2(val, iregs[i * 2 + 0][j * 2 + 1]);
            val       = __hadd2(val, iregs[i * 2 + 0][j * 2 + 2]);
            val       = __hadd2(val, iregs[i * 2 + 1][j * 2 + 0]);
            val       = __hadd2(val, iregs[i * 2 + 1][j * 2 + 1]);
            val       = __hadd2(val, iregs[i * 2 + 1][j * 2 + 2]);
            val       = __hadd2(val, iregs[i * 2 + 2][j * 2 + 0]);
            val       = __hadd2(val, iregs[i * 2 + 2][j * 2 + 1]);
            val       = __hadd2(val, iregs[i * 2 + 2][j * 2 + 2]);

            val.x = __hdiv(val.x, cnt);
            val.y = __hdiv(val.y, cnt);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s2_NHWC(
    const T* input,
    T* output,
    int if_exclude_padding,
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

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : T(0);
            iregs[i][j]   = ival;
        }
    }

    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val       = val + iregs[i * 2 + 0][j * 2 + 1];
            val       = val + iregs[i * 2 + 0][j * 2 + 2];
            val       = val + iregs[i * 2 + 1][j * 2 + 0];
            val       = val + iregs[i * 2 + 1][j * 2 + 1];
            val       = val + iregs[i * 2 + 1][j * 2 + 2];
            val       = val + iregs[i * 2 + 2][j * 2 + 0];
            val       = val + iregs[i * 2 + 2][j * 2 + 1];
            val       = val + iregs[i * 2 + 2][j * 2 + 2];

            val = val / cnt;

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_NHWC_int8(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
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
    int8_t iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            int8_t ival    = pred ? input[in_off + in_off_hw] : 0;
            iregs[i][j]   = ival;
        }
    }

    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i * 2 + 0][j * 2 + 0];
            val       = val + iregs[i * 2 + 0][j * 2 + 1];
            val       = val + iregs[i * 2 + 0][j * 2 + 2];
            val       = val + iregs[i * 2 + 1][j * 2 + 0];
            val       = val + iregs[i * 2 + 1][j * 2 + 1];
            val       = val + iregs[i * 2 + 1][j * 2 + 2];
            val       = val + iregs[i * 2 + 2][j * 2 + 0];
            val       = val + iregs[i * 2 + 2][j * 2 + 1];
            val       = val + iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = round((val / cnt) * in_scale / out_scale);
            }
        }
    }
}

// #################### pooling ave f3s1 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_half2_NHWC(
    const half2* input,
    half2* output,
    int if_exclude_padding,
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
            half2 ival    = pred ? input[in_off + in_off_hw] : half2(0.f, 0.f);
            iregs[i][j]   = ival;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half2 val = iregs[i + 0][j + 0];
            val       = __hadd2(val, iregs[i + 0][j + 1]);
            val       = __hadd2(val, iregs[i + 0][j + 2]);
            val       = __hadd2(val, iregs[i + 1][j + 0]);
            val       = __hadd2(val, iregs[i + 1][j + 1]);
            val       = __hadd2(val, iregs[i + 1][j + 2]);
            val       = __hadd2(val, iregs[i + 2][j + 0]);
            val       = __hadd2(val, iregs[i + 2][j + 1]);
            val       = __hadd2(val, iregs[i + 2][j + 2]);
            val.x     = __hdiv(val.x, cnt);
            val.y     = __hdiv(val.y, cnt);

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s1_NHWC(
    const T* input,
    T* output,
    int if_exclude_padding,
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

    // register blocking for input
    T iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : T(0);
            iregs[i][j]   = ival;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i + 0][j + 0];
            val       = val + iregs[i + 0][j + 1];
            val       = val + iregs[i + 0][j + 2];
            val       = val + iregs[i + 1][j + 0];
            val       = val + iregs[i + 1][j + 1];
            val       = val + iregs[i + 1][j + 2];
            val       = val + iregs[i + 2][j + 0];
            val       = val + iregs[i + 2][j + 1];
            val       = val + iregs[i + 2][j + 2];
            val       = val / cnt;

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_NHWC_int8(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
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
    int8_t iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            int8_t ival    = pred ? input[in_off + in_off_hw] : 0;
            iregs[i][j]   = ival;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i + 0][j + 0];
            val       = val + iregs[i + 0][j + 1];
            val       = val + iregs[i + 0][j + 2];
            val       = val + iregs[i + 1][j + 0];
            val       = val + iregs[i + 1][j + 1];
            val       = val + iregs[i + 1][j + 2];
            val       = val + iregs[i + 2][j + 0];
            val       = val + iregs[i + 2][j + 1];
            val       = val + iregs[i + 2][j + 2];

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = round((val / cnt) * in_scale / out_scale);
            }
        }
    }
}

// #################### pooling ave #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_half2_NHWC(
    const half2* input,
    half2* output,
    int if_exclude_padding,
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
            half2 res = half2(0.f, 0.f);
            int cnt   = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    half2 ival    = pred ? input[in_off + in_off_hw] : half2(0.f, 0.f);
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                    res = __hadd2(res, ival);
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            res.x = __hdiv(res.x, cnt);
            res.y = __hdiv(res.y, cnt);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_common_NHWC(
    const T* input,
    T* output,
    int if_exclude_padding,
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
            T res = T(0);
            int cnt   = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : T(0);
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                    res = res + ival;
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            res = res / cnt;

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_NHWC_int8(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
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
            int32_t res = 0;
            int cnt   = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    int8_t ival    = pred ? input[in_off + in_off_hw] : 0;
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                    res = res + ival;
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = round((res / cnt) * in_scale / out_scale);
            }
        }
    }
}


ppl::common::RetCode PPLCUDAAvePoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const half* input,
    ppl::nn::TensorShape* output_shape,
    half* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

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
            ppl_cukernel_pooling_ave_f3s1_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(input, output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(input, output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(input, output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
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
            ppl_cukernel_pooling_ave_f3s1_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, if_exclude_padding, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, if_exclude_padding, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common_half2_NHWC<4, 1><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const half2*)input, (half2*)output, if_exclude_padding, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

template<typename T>
ppl::common::RetCode PPLCUDAAvePoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const T* input,
    ppl::nn::TensorShape* output_shape,
    T* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

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
            ppl_cukernel_pooling_ave_f3s1<4, 1, T><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2<4, 1, T><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common<4, 1, T><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int partH             = (out_height + 3) / 4; //tile
        int partW             = (out_width + 0) / 1;
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1_NHWC<4, 1, T><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const T*)input, (T*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_NHWC<4, 1, T><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const T*)input, (T*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common_NHWC<4, 1, T><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const T*)input, (T*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAAvePoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const int8_t* input,
    ppl::nn::TensorShape* output_shape,
    int8_t* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding,
    float in_scale,
    float out_scale)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

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
            ppl_cukernel_pooling_ave_f3s1_int8<4, 1><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_int8<4, 1><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_ave_common_int8<4, 1><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int partH             = (out_height + 3) / 4; //tile
        int partW             = (out_width + 0) / 1;
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1_NHWC_int8<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_NHWC_int8<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_ave_common_NHWC_int8<4, 1><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAAvePoolingForwardImp(
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
    int if_exclude_padding,
    float in_scale,
    float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAAvePoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, if_exclude_padding);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAAvePoolingForwardImp<float>(
            stream, input_shape, (const float*)input, output_shape, (float*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, if_exclude_padding);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        return PPLCUDAAvePoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, if_exclude_padding, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}