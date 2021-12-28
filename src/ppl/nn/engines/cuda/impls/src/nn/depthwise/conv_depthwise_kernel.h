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

#ifndef PPL_CUKERNEL_CONV_DEPTHWISE_KERNEL_H_
#define PPL_CUKERNEL_CONV_DEPTHWISE_KERNEL_H_

#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/nn/conv_fuse_type.h"
#include "cudakernel/nn/conv/conv_fp16.h"

#include "cudakernel/common/cuda_arch.h"

#include <cuda_fp16.h>
#include <vector>


#define PARAMLIST      const half* input, \
    const half* filter,\
    const half* bias,\
    DivModFast padc_fast,\
    DivModFast hw_fast,\
    DivModFast width_fast,\
    int in_height,\
    int in_width,\
    int kernel_h,\
    int kernel_w,\
    int pad_height,\
    int pad_width,\
    int stride_height,\
    int stride_width,\
    int hole_h, \
    int hole_w,\
    int tile_height,\
    int tile_width,\
    int channels,\
    int paddingc,\
    int out_height,\
    int out_width,\
    int in_batch_stride,\
    int in_height_stride,\
    int in_width_stride,\
    int total_elements,\
    half* output,\
    fuse_param_t fuse_params


template<int TILE_H, int TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W, typename T>
__forceinline__ __device__  void load_weight(const T* filter, int c_idx, int paddingc, T flt_val[KERNEL_H][KERNEL_W]) {
    int offset = c_idx;
#pragma unroll
    for (int i = 0; i < KERNEL_H; i++) {
#pragma unroll
        for (int j = 0; j < KERNEL_W; j++) {
            flt_val[i][j] = filter[offset];
            offset += paddingc;
        }
    }
}


template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W, bool INSCOPE, typename T>
__forceinline__ __device__ void load_feature(const T* input, int base_offset, int paddingc,
                                    int idx_h, int idx_w, int in_height, int in_width,
                                    T pic_val[SRC_TILE_H][SRC_TILE_W]) {
    bool pred_height = true;
#pragma unroll
    for (int i = 0; i < SRC_TILE_H; i++) {
        pred_height = (idx_h + i < in_height) && (idx_h + i >= 0);
#pragma unroll
        for (int j = 0; j < SRC_TILE_W; j++) {
            bool pred_width = pred_height && (idx_w + j < in_width) && (idx_w + j >= 0);
            pic_val[i][j] = pred_width ? input[base_offset + i * in_width * paddingc + j * paddingc] : T(0.0f);
        }
    }
}


template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W, bool INSCOPE, typename T>
__forceinline__ __device__ void load_feature_inpic(const T* input, int base_offset, int paddingc,
                                    int idx_h, int idx_w, int in_height, int in_width,
                                    T pic_val[SRC_TILE_H][SRC_TILE_W]) {
#pragma unroll
    for (int i = 0; i < SRC_TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < SRC_TILE_W; j++) {
            pic_val[i][j] = input[base_offset + i * in_width * paddingc + j * paddingc];
        }
    }
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__forceinline__ __device__ void compute_half(half pic_val[SRC_TILE_H][(TILE_W - 1) * STRIRDE_W + KERNEL_W], half flt_val[KERNEL_H][KERNEL_W], half out_val[TILE_H][TILE_W]) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_H; j++) {
#pragma unroll
            for (int r = 0; r < KERNEL_H; r++) {
#pragma unroll
                for (int s = 0; s < KERNEL_W; s++) {
                    if (r == 0 && s == 0)
                        out_val[i][j] = __hfma(flt_val[0][0], pic_val[i * STRIRDE_H][j * STRIRDE_W], (half)0);
                    else
                        out_val[i][j] = __hfma(flt_val[r][s], pic_val[i * STRIRDE_H + r][j * STRIRDE_W + s], out_val[i][j]);
                }
            }
        }
    }
#endif
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__forceinline__ __device__ void compute_float(float pic_val[SRC_TILE_H][(TILE_W - 1) * STRIRDE_W + KERNEL_W], float flt_val[KERNEL_H][KERNEL_W], float out_val[TILE_H][TILE_W]) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
    for (int i = 0; i < TILE_H ; i++) {
#pragma unroll
        for (int j = 0; j < TILE_H; j++) {
#pragma unroll
            for(int r = 0; r < KERNEL_H; r++) {
#pragma unroll
                for (int s = 0; s < KERNEL_W; s++) {
                    if (r == 0 && s == 0) 
                        out_val[i][j] = flt_val[0][0] * pic_val[i * STRIRDE_H][j * STRIRDE_W];
                    else 
                        out_val[i][j] = flt_val[r][s] * pic_val[i * STRIRDE_H + r][j * STRIRDE_W + s] + out_val[i][j];
                }
            }
        }
    }
#endif
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__forceinline__ __device__ void compute_int8(int8_t pic_val[SRC_TILE_H][(TILE_W - 1) * STRIRDE_W + KERNEL_W], int8_t flt_val[KERNEL_H][KERNEL_W], float out_val[TILE_H][TILE_W]) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
    for (int i = 0; i < TILE_H ; i++) {
#pragma unroll
        for (int j = 0; j < TILE_H; j++) {
#pragma unroll
            for(int r = 0; r < KERNEL_H; r++) {
#pragma unroll
                for (int s = 0; s < KERNEL_W; s++) {
                    if (r == 0 && s == 0) 
                        out_val[i][j] = flt_val[0][0] * pic_val[i * STRIRDE_H][j * STRIRDE_W];
                    else 
                        out_val[i][j] = flt_val[r][s] * pic_val[i * STRIRDE_H + r][j * STRIRDE_W + s] + out_val[i][j];
                }
            }
        }
    }
#endif
}

template<int TILE_H, int TILE_W>
__forceinline__ __device__ void load_bias_half(
    const half* bias,
    int c_idx,
    int channels,
    half out_val[TILE_H][TILE_W])
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    half bias_val = c_idx < channels ? bias[c_idx] : (half)0.0;
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_W; j++) {
            out_val[i][j] = __hadd(out_val[i][j], bias_val);
        }
    }
#endif
}

template<int TILE_H, int TILE_W>
__forceinline__ __device__ void load_bias_float(
    const float* bias,
    int c_idx,
    int channels,
    float out_val[TILE_H][TILE_W]) 
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    float bias_val = c_idx < channels ? bias[c_idx] : 0.0;
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_W; j++) {
            out_val[i][j] = out_val[i][j] + bias_val;
        }
    }
#endif
}

template<int TILE_H, int TILE_W>
__forceinline__ __device__ void load_bias_int8(
    const float* bias,
    int c_idx,
    int channels,
    float out_val[TILE_H][TILE_W],
    const float pic_scale,
    const float* flt_scale) 
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    float bias_val = c_idx < channels ? bias[c_idx] : 0;
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_W; j++) {
            out_val[i][j] = out_val[i][j] * pic_scale * (c_idx < channels ? flt_scale[c_idx] : 0) + bias_val;
        }
    }
#endif
}

template<int TILE_H, int TILE_W>
__forceinline__ __device__ void fuse_process_half(
    half out_val[TILE_H][TILE_W],
    int h_idx,
    int w_idx,
    int c_idx,
    int out_height,
    int out_width,
    int channels,
    int paddingc,
    int base_offset,
    fuse_param_t fuse_params)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#if __CUDACC_VER_MAJOR__ != 11 && __CUDACC_VER_MINOR__ != 3
#pragma unroll
#endif
    for (int i = 0; i < TILE_H; i++) {
#if __CUDACC_VER_MAJOR__ != 11 && __CUDACC_VER_MINOR__ != 3
#pragma unroll
#endif
        for (int j = 0; j < TILE_W; j++) {
            if (fuse_params.has_activation) {
                if (fuse_params.has_activation == 1) {
                    out_val[i][j] = __hge(out_val[i][j], (half)0.0) ? out_val[i][j] : (half)0.0;
                } else { // if (fuse_params.has_activation == 2){
                    __half one    = (__half)1;
                    __half tmp    = hexp(out_val[i][j]);
                    out_val[i][j] = __hdiv(tmp, __hadd(one, tmp));
                }
            } else if (fuse_params.has_clip) {
                out_val[i][j] = __hge(out_val[i][j], __float2half(fuse_params.clip_max)) ? __float2half(fuse_params.clip_max) : __hle(out_val[i][j], __float2half(fuse_params.clip_min)) ? __float2half(fuse_params.clip_min)
                                                                                                                                                                                         : out_val[i][j];
            } else if (fuse_params.has_prelu && __hlt(out_val[i][j], 0)) {
                if (fuse_params.has_prelu == 1) {
                    out_val[i][j] = __hmul(out_val[i][j], __float2half(fuse_params.leaky));
                } else if (fuse_params.has_prelu == 2) {
                    out_val[i][j] = __hmul(out_val[i][j], ((half*)fuse_params.prelu)[c_idx]);
                } else if (fuse_params.has_prelu == 3) {
                    out_val[i][j] = __hmul(out_val[i][j], ((half*)fuse_params.elt_prelu)[base_offset + i * out_width * paddingc + j * paddingc]);
                }
            }
            if (fuse_params.has_elt) {
                bool in_padding = h_idx * TILE_H + i < out_height && w_idx * TILE_W + j < out_width;

                if (in_padding)
                    out_val[i][j] = __hadd(out_val[i][j], ((half*)fuse_params.pre_data)[base_offset + i * out_width * paddingc + j * paddingc]);
                if (fuse_params.has_elt_activation) {
                    if (fuse_params.has_elt_activation == 1) {
                        out_val[i][j] = __hge(out_val[i][j], (half)0.0) ? out_val[i][j] : (half)0.0;
                    } else { // if (fuse_params.has_activation == 2){
                        __half one    = (__half)1;
                        __half tmp    = hexp(out_val[i][j]);
                        out_val[i][j] = __hdiv(tmp, __hadd(one, tmp));
                    }
                } else if (fuse_params.has_elt_clip) {
                    out_val[i][j] = __hge(out_val[i][j], __float2half(fuse_params.elt_clip_max)) ? __float2half(fuse_params.elt_clip_max) : __hle(out_val[i][j], __float2half(fuse_params.elt_clip_min)) ? __float2half(fuse_params.elt_clip_min)
                                                                                                                                                                                                         : out_val[i][j];
                } else if (fuse_params.has_elt_prelu && __hlt(out_val[i][j], 0)) {
                    out_val[i][j] = __hmul(out_val[i][j], ((half*)fuse_params.elt_prelu)[c_idx]);
                    if (fuse_params.has_prelu == 1) {
                        out_val[i][j] = __hmul(out_val[i][j], __float2half(fuse_params.elt_leaky));
                    } else if (fuse_params.has_prelu == 2) {
                        out_val[i][j] = __hmul(out_val[i][j], ((half*)fuse_params.elt_prelu)[c_idx]);
                    } else if (fuse_params.has_prelu == 3) {
                        out_val[i][j] = __hmul(out_val[i][j], ((half*)fuse_params.elt_prelu)[base_offset + i * out_width * paddingc + j * paddingc]);
                    }
                }
            }
        }
    }
#endif
}

template<int TILE_H, int TILE_W>
__forceinline__ __device__ void fuse_process_float(
    float out_val[TILE_H][TILE_W],
    int h_idx,
    int w_idx,
    int c_idx,
    int out_height, 
    int out_width,
    int channels,
    int paddingc,
    int base_offset,
    fuse_param_t fuse_params
)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_W; j++) {
            if (fuse_params.has_activation){
                if (fuse_params.has_activation == 1){
                    out_val[i][j] = out_val[i][j] >= 0.0f ? out_val[i][j] : 0.0f;
		        }else{// if (fuse_params.has_activation == 2){
                    float tmp = exp(out_val[i][j]);
                    out_val[i][j] = tmp / (tmp + 1);
		        }
            } else if (fuse_params.has_clip) {
                out_val[i][j] = out_val[i][j] >= fuse_params.clip_max ? 
                        fuse_params.clip_max : out_val[i][j] <= fuse_params.clip_min ? 
                        fuse_params.clip_min : out_val[i][j];
            } else if (fuse_params.has_prelu && out_val[i][j] < 0 ) {
                if(fuse_params.has_prelu == 1){
                            out_val[i][j] = out_val[i][j] * fuse_params.leaky;
                } else if(fuse_params.has_prelu == 2){
                            out_val[i][j] = out_val[i][j] * ((float*)fuse_params.prelu)[c_idx];
                } else if(fuse_params.has_prelu == 3){
                            out_val[i][j] = out_val[i][j] * ((float*)fuse_params.elt_prelu)[base_offset + i * out_width * paddingc + j * paddingc];
                }
            }
            if (fuse_params.has_elt) {
                bool in_padding = h_idx * TILE_H + i < out_height && w_idx * TILE_W + j < out_width;

                if (in_padding) out_val[i][j] = out_val[i][j] + ((float*)fuse_params.pre_data)[base_offset + i * out_width * paddingc + j * paddingc];
                if (fuse_params.has_elt_activation){
                    if (fuse_params.has_elt_activation == 1){
                        out_val[i][j] = out_val[i][j] >= 0.0f ? out_val[i][j] : 0.0f;
                    }else{// if (fuse_params.has_activation == 2){
                        float tmp = exp(out_val[i][j]);
                        out_val[i][j] = tmp / (tmp + 1);
                    }
                } else if (fuse_params.has_elt_clip) {
                    out_val[i][j] = out_val[i][j] >= fuse_params.elt_clip_max ? 
                            fuse_params.elt_clip_max : out_val[i][j] <= fuse_params.elt_clip_min ? 
                            fuse_params.elt_clip_min : out_val[i][j];
                } else if (fuse_params.has_elt_prelu && out_val[i][j] < 0 ) {
                    out_val[i][j] = out_val[i][j] * ((float*)fuse_params.elt_prelu)[c_idx];
                    if(fuse_params.has_elt_prelu == 1){
                        out_val[i][j] = out_val[i][j] * fuse_params.elt_leaky;
                    } else if(fuse_params.has_elt_prelu == 2){
                        out_val[i][j] = out_val[i][j] * ((float*)fuse_params.elt_prelu)[c_idx];
                    } else if(fuse_params.has_elt_prelu == 3){
                        out_val[i][j] = out_val[i][j] * ((float*)fuse_params.elt_prelu)[base_offset + i * out_width * paddingc + j * paddingc];
                    }
                }                
            }
        }
    }
#endif  
}



template<int TILE_H, int TILE_W, typename T>
__forceinline__ __device__ void write_global(
    T out_val[TILE_H][TILE_W],
    int h_idx,
    int w_idx,
    int c_idx,
    int out_height,
    int out_width,
    int channels,
    int paddingc,
    int base_offset,
    T* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_W; j++) {
            bool in_padding = h_idx * TILE_H + i < out_height && w_idx * TILE_W + j < out_width;
            if (in_padding)
	        {
                output[base_offset + i * out_width * paddingc + j * paddingc] = c_idx < channels ? out_val[i][j] : (T)0.0f;
            }
        }
    }
#endif
}

template<int TILE_H, int TILE_W>
__forceinline__ __device__ void write_global_int8(
    float out_val[TILE_H][TILE_W],
    int h_idx,
    int w_idx,
    int c_idx,
    int out_height, 
    int out_width,
    int channels,
    int paddingc,
    int base_offset,
    int8_t* output,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll
        for (int j = 0; j < TILE_W; j++) {
            bool in_padding = h_idx * TILE_H + i < out_height && w_idx * TILE_W + j < out_width;
            if (in_padding) {
                int32_t res = round(out_val[i][j] * out_scale);
                if(res > 127) res = 127;
                else if(res < -128) res = -128;
                output[base_offset + i * out_width * paddingc + j * paddingc] = c_idx < channels ? res : 0;
            }
        }
    }
#endif
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_fmma(
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
    fuse_param_t fuse_params) 
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int tile_nhw_idx, c_idx;
    padc_fast.divmod(tid, tile_nhw_idx, c_idx);
    int tile_hw_idx = hw_fast.mod(tile_nhw_idx);
    int h_idx, w_idx;
    width_fast.divmod(tile_hw_idx, h_idx, w_idx);
    int n_idx = tid / (paddingc * tile_height * tile_width);

    float out_val[TILE_H][TILE_W];
    float flt_val[KERNEL_H][KERNEL_W];
    
    float pic_val[SRC_TILE_H][SRC_TILE_W];
    load_weight<TILE_H, TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, float>(filter, c_idx, paddingc, flt_val); 

    int in_w_idx = (w_idx * TILE_W) * STRIRDE_W - pad_width;
    int in_h_idx = (h_idx * TILE_H) * STRIRDE_H - pad_height;

    bool src_in_pic = (in_w_idx >= 0) && (in_w_idx + SRC_TILE_W < in_width) && (in_h_idx >= 0) && (in_h_idx + SRC_TILE_H < in_height);
    int base_offset = n_idx * in_batch_stride + in_h_idx * in_height_stride + in_w_idx * in_width_stride + c_idx;

    if (src_in_pic) {
        load_feature_inpic<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, true, float>(
                    input, base_offset, paddingc, in_h_idx, in_w_idx, in_height, in_width, pic_val);
    } else {
        load_feature<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, false, float>(
                    input, base_offset, paddingc, in_h_idx, in_w_idx, in_height, in_width, pic_val);
    }
    compute_float<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W>(pic_val, flt_val, out_val);
    base_offset = n_idx * out_height * out_width * paddingc + h_idx  * TILE_H * out_width * paddingc + w_idx * TILE_W * paddingc + c_idx;

    if (bias) {
        load_bias_float<TILE_H, TILE_W>(bias, c_idx, channels, out_val);
    }
    
    fuse_process_float<TILE_H, TILE_W>(out_val, h_idx, w_idx, c_idx, out_height, out_width, channels, paddingc, base_offset, fuse_params);
    if (fuse_params.has_concat) {
        paddingc = fuse_params.concat_stride;
        base_offset = fuse_params.concat_offset + n_idx * out_height * out_width * paddingc + h_idx  * TILE_H * out_width * paddingc + w_idx * TILE_W * paddingc + c_idx;
        output = (float*)fuse_params.post_concat;
    }
    write_global<TILE_H, TILE_W, float>(out_val, h_idx, w_idx, c_idx, out_height, out_width, channels, paddingc, base_offset, output);
#endif
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_int8mma(
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
    fuse_param_t fuse_params,
    const float pic_scale,
    const float* flt_scale,
    const float out_scale) 
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int tile_nhw_idx, c_idx;
    padc_fast.divmod(tid, tile_nhw_idx, c_idx);
    int tile_hw_idx = hw_fast.mod(tile_nhw_idx);
    int h_idx, w_idx;
    width_fast.divmod(tile_hw_idx, h_idx, w_idx);
    int n_idx = tid / (paddingc * tile_height * tile_width);

    float out_val[TILE_H][TILE_W];
    int8_t flt_val[KERNEL_H][KERNEL_W];
    
    int8_t pic_val[SRC_TILE_H][SRC_TILE_W];
    load_weight<TILE_H, TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, int8_t>(filter, c_idx, paddingc, flt_val); 

    int in_w_idx = (w_idx * TILE_W) * STRIRDE_W - pad_width;
    int in_h_idx = (h_idx * TILE_H) * STRIRDE_H - pad_height;

    bool src_in_pic = (in_w_idx >= 0) && (in_w_idx + SRC_TILE_W < in_width) && (in_h_idx >= 0) && (in_h_idx + SRC_TILE_H < in_height);
    int base_offset = n_idx * in_batch_stride + in_h_idx * in_height_stride + in_w_idx * in_width_stride + c_idx;

    if (src_in_pic) {
        load_feature_inpic<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, true, int8_t>(
                    input, base_offset, paddingc, in_h_idx, in_w_idx, in_height, in_width, pic_val);
    } else {
        load_feature<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, false, int8_t>(
                    input, base_offset, paddingc, in_h_idx, in_w_idx, in_height, in_width, pic_val);
    }
    compute_int8<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W>(pic_val, flt_val, out_val);
    base_offset = n_idx * out_height * out_width * paddingc + h_idx  * TILE_H * out_width * paddingc + w_idx * TILE_W * paddingc + c_idx;

    if (bias) {
        load_bias_int8<TILE_H, TILE_W>(bias, c_idx, channels, out_val, pic_scale, flt_scale);
    }
    
    fuse_process_float<TILE_H, TILE_W>(out_val, h_idx, w_idx, c_idx, out_height, out_width, channels, paddingc, base_offset, fuse_params);
    if (fuse_params.has_concat) {
        paddingc = fuse_params.concat_stride;
        base_offset = fuse_params.concat_offset + n_idx * out_height * out_width * paddingc + h_idx  * TILE_H * out_width * paddingc + w_idx * TILE_W * paddingc + c_idx;
        output = (int8_t*)fuse_params.post_concat;
    }
    
    write_global_int8<TILE_H, TILE_W>(out_val, h_idx, w_idx, c_idx, out_height, out_width, channels, paddingc, base_offset, output, out_scale);
#endif
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_hmma(
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
    fuse_param_t fuse_params)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements)
        return;

    int tile_nhw_idx, c_idx;
    padc_fast.divmod(tid, tile_nhw_idx, c_idx);
    int tile_hw_idx = hw_fast.mod(tile_nhw_idx);
    int h_idx, w_idx;
    width_fast.divmod(tile_hw_idx, h_idx, w_idx);
    int n_idx = tid / (paddingc * tile_height * tile_width);

    half out_val[TILE_H][TILE_W];
    half flt_val[KERNEL_H][KERNEL_W];

    half pic_val[SRC_TILE_H][SRC_TILE_W];
    load_weight<TILE_H, TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W>(filter, c_idx, paddingc, flt_val);

    int in_w_idx = (w_idx * TILE_W) * STRIRDE_W - pad_width;
    int in_h_idx = (h_idx * TILE_H) * STRIRDE_H - pad_height;

    bool src_in_pic = (in_w_idx >= 0) && (in_w_idx + SRC_TILE_W < in_width) && (in_h_idx >= 0) && (in_h_idx + SRC_TILE_H < in_height);
    int base_offset = n_idx * in_batch_stride + in_h_idx * in_height_stride + in_w_idx * in_width_stride + c_idx;

    if (src_in_pic) {
        load_feature_inpic<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, true>(
            input, base_offset, paddingc, in_h_idx, in_w_idx, in_height, in_width, pic_val);
    } else {
        load_feature<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W, false>(
            input, base_offset, paddingc, in_h_idx, in_w_idx, in_height, in_width, pic_val);
    }
    compute_half<TILE_H, TILE_W, SRC_TILE_H, SRC_TILE_W, KERNEL_H, KERNEL_W, STRIRDE_H, STRIRDE_W>(pic_val, flt_val, out_val);
    base_offset = n_idx * out_height * out_width * paddingc + h_idx  * TILE_H * out_width * paddingc + w_idx * TILE_W * paddingc + c_idx;

    if (bias) {
        load_bias_half<TILE_H, TILE_W>(bias, c_idx, channels, out_val);
    }

    fuse_process_half<TILE_H, TILE_W>(out_val, h_idx, w_idx, c_idx, out_height, out_width, channels, paddingc, base_offset, fuse_params);
    if (fuse_params.has_concat) {
        paddingc    = fuse_params.concat_stride;
        base_offset = fuse_params.concat_offset + n_idx * out_height * out_width * paddingc + h_idx * TILE_H * out_width * paddingc + w_idx * TILE_W * paddingc + c_idx;
        output      = (half*)fuse_params.post_concat;
    }
    write_global<TILE_H, TILE_W, half>(out_val, h_idx, w_idx, c_idx, out_height, out_width, channels, paddingc, base_offset, output);
#endif
}

template<>
__global__ void ppl_cuda_depthwise_fmma<-1,-1,-1,-1,-1,-1,-1,-1>(
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
    fuse_param_t fuse_params) 
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int tile_nhw_idx, c_idx;
    padc_fast.divmod(tid, tile_nhw_idx, c_idx);

    int tile_hw_idx = hw_fast.mod(tile_nhw_idx);

    int h_idx, w_idx;
    width_fast.divmod(tile_hw_idx, h_idx, w_idx);

    int n_idx = tid / (paddingc * tile_height * tile_width);
    float out_val = 0.0f;
    float flt_val;
    float pic_val;
    
    int in_w_idx = w_idx * stride_width - pad_width;
    int in_h_idx = h_idx * stride_height - pad_height;

    int base_offset = n_idx * in_batch_stride + in_h_idx * in_height_stride + in_w_idx * in_width_stride + c_idx;
    int flt_offset = c_idx;
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            flt_val = filter[flt_offset];
            flt_offset += paddingc;
            bool pred = (in_h_idx + i * hole_h >= 0) && (in_h_idx + i * hole_h < in_height) && (in_w_idx + j * hole_w >= 0) && (in_w_idx + j * hole_w < in_width);
            pic_val = pred ? input[base_offset + i * hole_h * in_height_stride + j * hole_w * in_width_stride] : 0; 
            out_val = flt_val * pic_val + out_val;
        }
    }

    base_offset = n_idx * out_height * out_width * paddingc + h_idx * out_width * paddingc + w_idx * paddingc + c_idx;

    if (bias) {
        float bias_val = c_idx < channels ? bias[c_idx] : 0.0f; 
        out_val = out_val + bias_val;
    }
    if (fuse_params.has_activation){
        if (fuse_params.has_activation == 1 ){
            out_val = out_val >= 0.0f ? out_val : 0.0f;
        } else{// if (fuse_params.has_activation == 2 ){
            float tmp = exp(out_val);
            out_val = tmp / (tmp + 1);
        }
    } else if (fuse_params.has_clip) {
        out_val = out_val >= fuse_params.clip_max ? 
                fuse_params.clip_max : out_val <= fuse_params.clip_min ? 
                fuse_params.clip_min : out_val;
    } else if (fuse_params.has_prelu) {
        out_val = out_val * ((float*)fuse_params.prelu)[c_idx];
    }
    if (fuse_params.has_elt) {
        out_val = out_val + ((float*)fuse_params.pre_data)[base_offset];
        if (fuse_params.has_elt_activation){
            if (fuse_params.has_elt_activation == 1){
                out_val = out_val >= 0.0f ? out_val : 0.0f;
            } else{// if (fuse_params.has_activation == 2 ){
                float tmp = exp(out_val);
                out_val = tmp / (tmp + 1);
            }
        } else if (fuse_params.has_elt_clip) {
            out_val = out_val >= fuse_params.elt_clip_max ? 
                    fuse_params.elt_clip_max : out_val <= fuse_params.elt_clip_min ? 
                    fuse_params.elt_clip_min : out_val;
        } else if (fuse_params.has_elt_prelu) {
            out_val = out_val * ((float*)fuse_params.elt_prelu)[c_idx];
        }                
    }
    if (fuse_params.has_concat) {
        output = (float*)fuse_params.post_concat;
        paddingc = fuse_params.concat_stride;
        base_offset = fuse_params.concat_offset + n_idx * out_height * out_width * paddingc + h_idx * out_width * paddingc + w_idx * paddingc + c_idx;
    }
    output[base_offset] = out_val;
#endif
}

template <>
__global__ void ppl_cuda_depthwise_hmma<-1, -1, -1, -1, -1, -1, -1, -1>(
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
    fuse_param_t fuse_params)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements)
        return;

    int tile_nhw_idx, c_idx;
    padc_fast.divmod(tid, tile_nhw_idx, c_idx);

    int tile_hw_idx = hw_fast.mod(tile_nhw_idx);

    int h_idx, w_idx;
    width_fast.divmod(tile_hw_idx, h_idx, w_idx);

    int n_idx    = tid / (paddingc * tile_height * tile_width);
    half out_val = (half)0.0;
    half flt_val;
    half pic_val;

    int in_w_idx = w_idx * stride_width - pad_width;
    int in_h_idx = h_idx * stride_height - pad_height;

    int base_offset = n_idx * in_batch_stride + in_h_idx * in_height_stride + in_w_idx * in_width_stride + c_idx;
    int flt_offset  = c_idx;
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            flt_val = filter[flt_offset];
            flt_offset += paddingc;
            bool pred = (in_h_idx + i * hole_h >= 0) && (in_h_idx + i * hole_h < in_height) && (in_w_idx + j * hole_w >= 0) && (in_w_idx + j * hole_w < in_width);
            pic_val   = pred ? input[base_offset + i * hole_h * in_height_stride + j * hole_w * in_width_stride] : __float2half(0.0f);
            out_val   = __hfma(flt_val, pic_val, out_val);
        }
    }

    base_offset = n_idx * out_height * out_width * paddingc + h_idx * out_width * paddingc + w_idx * paddingc + c_idx;

    if (bias) {
        half bias_val = c_idx < channels ? bias[c_idx] : __float2half(0.0f);
        out_val       = __hadd(out_val, bias_val);
    }
    if (fuse_params.has_activation) {
        if (fuse_params.has_activation == 1) {
            out_val = __hge(out_val, (half)0.0) ? out_val : (half)0.0;
        } else { // if (fuse_params.has_activation == 2 ){
            __half one = (__half)1;
            __half tmp = hexp(out_val);
            out_val    = __hdiv(tmp, __hadd(one, tmp));
        }
    } else if (fuse_params.has_clip) {
        out_val = __hge(out_val, __float2half(fuse_params.clip_max)) ? __float2half(fuse_params.clip_max) : __hle(out_val, __float2half(fuse_params.clip_min)) ? __float2half(fuse_params.clip_min)
                                                                                                                                                               : out_val;
    } else if (fuse_params.has_prelu) {
        out_val = __hmul(out_val, ((half*)fuse_params.prelu)[c_idx]);
    }
    if (fuse_params.has_elt) {
        out_val = __hadd(out_val, ((half*)fuse_params.pre_data)[base_offset]);
        if (fuse_params.has_elt_activation) {
            if (fuse_params.has_elt_activation == 1) {
                out_val = __hge(out_val, (half)0.0) ? out_val : (half)0.0;
            } else { // if (fuse_params.has_activation == 2 ){
                __half one = (__half)1;
                __half tmp = hexp(out_val);
                out_val    = __hdiv(tmp, __hadd(one, tmp));
            }
        } else if (fuse_params.has_elt_clip) {
            out_val = __hge(out_val, __float2half(fuse_params.elt_clip_max)) ? __float2half(fuse_params.elt_clip_max) : __hle(out_val, __float2half(fuse_params.elt_clip_min)) ? __float2half(fuse_params.elt_clip_min)
                                                                                                                                                                               : out_val;
        } else if (fuse_params.has_elt_prelu) {
            out_val = __hmul(out_val, ((half*)fuse_params.elt_prelu)[c_idx]);
        }
    }
    if (fuse_params.has_concat) {
        output      = (half*)fuse_params.post_concat;
        paddingc    = fuse_params.concat_stride;
        base_offset = fuse_params.concat_offset + n_idx * out_height * out_width * paddingc + h_idx * out_width * paddingc + w_idx * paddingc + c_idx;
    }
    output[base_offset] = out_val;
#endif
}

template<>
__global__ void ppl_cuda_depthwise_int8mma<-1,-1,-1,-1,-1,-1,-1,-1>(
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
    fuse_param_t fuse_params,
    const float pic_scale,
    const float* flt_scale,
    const float out_scale) 
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int tile_nhw_idx, c_idx;
    padc_fast.divmod(tid, tile_nhw_idx, c_idx);

    int tile_hw_idx = hw_fast.mod(tile_nhw_idx);

    int h_idx, w_idx;
    width_fast.divmod(tile_hw_idx, h_idx, w_idx);

    int n_idx = tid / (paddingc * tile_height * tile_width);
    int32_t out_val0 = 0;
    int8_t flt_val;
    int8_t pic_val;
    
    int in_w_idx = w_idx * stride_width - pad_width;
    int in_h_idx = h_idx * stride_height - pad_height;

    int base_offset = n_idx * in_batch_stride + in_h_idx * in_height_stride + in_w_idx * in_width_stride + c_idx;
    int flt_offset = c_idx;
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            flt_val = filter[flt_offset];
            flt_offset += paddingc;
            bool pred = (in_h_idx + i * hole_h >= 0) && (in_h_idx + i * hole_h < in_height) && (in_w_idx + j * hole_w >= 0) && (in_w_idx + j * hole_w < in_width);
            pic_val = pred ? input[base_offset + i * hole_h * in_height_stride + j * hole_w * in_width_stride] : 0; 
            out_val0 = flt_val * pic_val + out_val0;
        }
    }

    base_offset = n_idx * out_height * out_width * paddingc + h_idx * out_width * paddingc + w_idx * paddingc + c_idx;
    float out_val = out_val0 * pic_scale * (c_idx < channels ? flt_scale[c_idx] : 0.0f);
    if (bias) {
        float bias_val = c_idx < channels ? bias[c_idx] : 0.0f; 
        out_val = out_val + bias_val;
    }
    if (fuse_params.has_activation){
        if (fuse_params.has_activation == 1 ){
            out_val = out_val >= 0.0f ? out_val : 0.0f;
        } else{// if (fuse_params.has_activation == 2 ){
            float tmp = exp(out_val);
            out_val = tmp / (tmp + 1);
        }
    } else if (fuse_params.has_clip) {
        out_val = out_val >= fuse_params.clip_max ? 
                fuse_params.clip_max : out_val <= fuse_params.clip_min ? 
                fuse_params.clip_min : out_val;
    } else if (fuse_params.has_prelu) {
        out_val = out_val * ((float*)fuse_params.prelu)[c_idx];
    }
    if (fuse_params.has_elt) {
        out_val = out_val + ((float*)fuse_params.pre_data)[base_offset];
        if (fuse_params.has_elt_activation){
            if (fuse_params.has_elt_activation == 1){
                out_val = out_val >= 0.0f ? out_val : 0.0f;
            } else{// if (fuse_params.has_activation == 2 ){
                float tmp = exp(out_val);
                out_val = tmp / (tmp + 1);
            }
        } else if (fuse_params.has_elt_clip) {
            out_val = out_val >= fuse_params.elt_clip_max ? 
                    fuse_params.elt_clip_max : out_val <= fuse_params.elt_clip_min ? 
                    fuse_params.elt_clip_min : out_val;
        } else if (fuse_params.has_elt_prelu) {
            out_val = out_val * ((float*)fuse_params.elt_prelu)[c_idx];
        }                
    }
    if (fuse_params.has_concat) {
        output = (int8_t*)fuse_params.post_concat;
        paddingc = fuse_params.concat_stride;
        base_offset = fuse_params.concat_offset + n_idx * out_height * out_width * paddingc + h_idx * out_width * paddingc + w_idx * paddingc + c_idx;
    }
    int32_t res = round(out_val * out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    output[base_offset] = res;
#endif
}

template<typename T>
__global__ void __launch_bounds__(32) ppl_cukernel_matrix_transpose(
    const T* filter,
    T* cvt_filter,
    int in_height,
    int in_width,
    int out_height,
    int out_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    __shared__ T shared[32][33];
    int tid          = threadIdx.x;
    int bx           = blockIdx.x;
    int by           = blockIdx.y;
    int bz           = blockIdx.z;
    int in_h         = by * 32;
    int in_w         = bx * 32 + tid;
    int out_h        = bx * 32;
    int out_w        = by * 32 + tid;
    uint64_t in_idx  = ((uint64_t)bz * in_height * in_width) + ((uint64_t)by * in_width * 32) + (bx * 32) + tid;
    uint64_t out_idx = ((uint64_t)bz * out_height * out_width) +
                       ((uint64_t)bx * out_width * 32) + (by * 32) + tid;

    if (in_w < in_width) {
        for (int i = 0; i < 32; i++) {
            if (in_h + i < in_height) {
                shared[i][tid] = filter[in_idx + i * in_width];
            }
        }
    }
    __syncthreads();

    T regz = (T)0.0;
    if (out_w < out_width) {
        for (int i = 0; i < 32; i++) {
            if (out_h + i < out_height) {
                cvt_filter[out_idx + i * out_width] = (out_w < in_height && (out_h + i) < in_width) ? shared[tid][i] : regz;
            }
        }
    }
#endif
}

__global__ void __launch_bounds__(32) ppl_cukernel_matrix_transpose_int8(
    const int8_t *filter,
    int8_t *cvt_filter,
    int in_height,
    int in_width,
    int out_height,
    int out_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9    

    __shared__ int8_t shared[32][33];
    int tid         = threadIdx.x;
    int bx          = blockIdx.x;
    int by          = blockIdx.y;
    int bz          = blockIdx.z;
    int in_h         = by * 32;
    int in_w         = bx * 32 + tid;
    int out_h        = bx * 32;
    int out_w        = by * 32 + tid;
    uint64_t in_idx  = ((uint64_t)bz * in_height * in_width) + ((uint64_t)by * in_width * 32) + (bx * 32) + tid;
    uint64_t out_idx = ((uint64_t)bz * out_height * out_width) +
                      ((uint64_t)bx * out_width * 32) + (by * 32) + tid;

    if (in_w < in_width) {
        for (int i = 0; i < 32; i++) {
            if (in_h + i < in_height) {
                shared[i][tid] = filter[in_idx + i * in_width];
            }
        }
    }
    __syncthreads();

    int8_t regz = 0;
    if (out_w < out_width) {
        for (int i = 0; i < 32; i++) {
            if (out_h + i < out_height) {
                if(out_w < in_height && (out_h + i) < in_width) {
                    cvt_filter[out_idx + i * out_width] = (out_w < in_height && (out_h + i) < in_width) ? shared[tid][i] : regz;
                }
            }
        }
    }
#endif
}

// template __global__ void ppl_cuda_depthwise_hmma<2,2,2,2,1,1,1,1>(PARAMLIST);
// template __global__ void ppl_cuda_depthwise_hmma<4,4,4,4,1,1,1,1>(PARAMLIST);
// template __global__ void ppl_cuda_depthwise_hmma<2,2,4,4,3,3,1,1>(PARAMLIST);
// template __global__ void ppl_cuda_depthwise_hmma<4,4,6,6,3,3,1,1>(PARAMLIST);
// template __global__ void ppl_cuda_depthwise_hmma<2,2,5,5,3,3,2,2>(PARAMLIST);
// template __global__ void ppl_cuda_depthwise_hmma<2,2,6,6,5,5,1,1>(PARAMLIST);

#endif // PPL_CUKERNEL_CONV_DEPTHWISE_KERNEL_COMMON_CUH_
