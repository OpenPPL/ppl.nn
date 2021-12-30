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

#ifndef PPL_CUKERNEL_CONV_SP_DEPTHWISE_KERNEL_H_
#define PPL_CUKERNEL_CONV_SP_DEPTHWISE_KERNEL_H_
#include <cuda_fp16.h>

#include "cudakernel/common/macro.h"

#define PREDEFINEIF #if
#define PREDEFINEENDIF #endif
#define PRAGMAUNROLL #pragma

#define FUSE_PROCESS_FLOAT(out_val, fuse_params, TILE_H, TILE_W, base_offset) \
{ \
    if (fuse_params.has_activation){ \
        if (fuse_params.has_activation == 1) \
            { \
            _Pragma("unroll")\
                for (int i = 0; i < TILE_H * TILE_W; i++) { \
                    out_val[i] =  out_val[i] >= 0.0f ? out_val[i] : 0.0f;\
		        }\
            }\
    } else if (fuse_params.has_clip) { \
            _Pragma("unroll")\
                for (int i = 0; i < TILE_H * TILE_W; i++) { \
                 out_val[i] =  out_val[i] >= fuse_params.clip_max ? \
                        fuse_params.clip_max :  out_val[i] <= fuse_params.clip_min ? \
                        fuse_params.clip_min :  out_val[i];\
            }\
    } else if (fuse_params.has_prelu) { \
        _Pragma("unroll") \
        for(int i = 0; i < TILE_H * TILE_W; i++) { \
            if (fuse_params.has_prelu == 1) { \
                out_val[i] = out_val[i] > 0 ? out_val[i] : out_val[i] * fuse_params.leaky; \
            } else if (fuse_params.has_prelu == 2) { \
                 out_val[i] = out_val[i] > 0 ? out_val[i] : out_val[i] * ((float*)fuse_params.prelu)[channel_idx + i]; \
            } else if (fuse_params.has_prelu == 2) { \
                 out_val[i] = out_val[i] > 0 ? out_val[i] : out_val[i] * ((float*)fuse_params.prelu)[base_offset + i]; \
            }\
        } \
    } \
}

#define PACK4INT(data, x, y, z, w) \
    x = (0xffu & x); \
    y = (0xffu & y) << 8;\
    z = (0xffu & z) << 16;\
    w = (0xffu & w) << 24;\
    data = x | y | z | w;

#define DEFINE_POS(x, y, z, w) \
    const int x = 0x000000ff; \
    const int y = 0x0000ff00; \
    const int z = 0x00ff0000; \
    const int w = 0xff000000; 

#define GET_POS4_VAL(val, data) \
    val.x = data & posX; \
    val.y = data & posY; \
    val.z = data & posZ; \
    val.w = data & posW;

#define COMPUTE_F3S1_POS0(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[0], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[0], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[0], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[0], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[1], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[1], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[1], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[1], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[2], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[2], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[2], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[2], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[3], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[3], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[3], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[3], tmp_val.w, C[15]);
#define COMPUTE_F3S1_POS1(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[1], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[1], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[1], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[1], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[2], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[2], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[2], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[2], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[3], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[3], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[3], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[3], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[4], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[4], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[4], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[4], tmp_val.w, C[15]); 

#define COMPUTE_F3S1_POS2(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[2], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[2], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[2], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[2], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[3], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[3], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[3], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[3], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[4], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[4], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[4], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[4], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[5], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[5], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[5], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[5], tmp_val.w, C[15]); 

#define FETCH_INPUT(pic_data, int_input) \
    pic_data[0] = pred[0] ? int_input[in_idx + 0*inh_stride] : 0;\
    pic_data[1] = pred[1] ? int_input[in_idx + 1*inh_stride] : 0;\
    pic_data[2] = pred[2] ? int_input[in_idx + 2*inh_stride] : 0;\
    pic_data[3] = pred[3] ? int_input[in_idx + 3*inh_stride] : 0;\
    pic_data[4] = pred[4] ? int_input[in_idx + 4*inh_stride] : 0;\
    pic_data[5] = pred[5] ? int_input[in_idx + 5*inh_stride] : 0;

#define FETCH_FILTER(flt_data, int_filter, i, j, k) \
    flt_data[0] = int_filter[flt_off + i*c_stride]; \
    flt_data[1] = int_filter[flt_off + j*c_stride]; \
    flt_data[2] = int_filter[flt_off + k*c_stride]; 

#define FETCH_INPUT_NINE(pic_data, int_input) \
    pic_data[0] = pred[0] ? int_input[in_idx + 0*inh_stride] : 0;\
    pic_data[1] = pred[1] ? int_input[in_idx + 1*inh_stride] : 0;\
    pic_data[2] = pred[2] ? int_input[in_idx + 2*inh_stride] : 0;\
    pic_data[3] = pred[3] ? int_input[in_idx + 3*inh_stride] : 0;\
    pic_data[4] = pred[4] ? int_input[in_idx + 4*inh_stride] : 0;\
    pic_data[5] = pred[5] ? int_input[in_idx + 5*inh_stride] : 0;\
    pic_data[6] = pred[6] ? int_input[in_idx + 6*inh_stride] : 0;\
    pic_data[7] = pred[7] ? int_input[in_idx + 7*inh_stride] : 0;\
    pic_data[8] = pred[8] ? int_input[in_idx + 8*inh_stride] : 0;



#define COMPUTE_F3S2_POS0(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[0], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[0], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[0], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[0], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[2], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[2], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[2], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[2], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[4], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[4], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[4], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[4], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[6], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[6], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[6], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[6], tmp_val.w, C[15]); 

#define COMPUTE_F3S2_POS1(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[1], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[1], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[1], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[1], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[3], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[3], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[3], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[3], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[5], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[5], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[5], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[5], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[7], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[7], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[7], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[7], tmp_val.w, C[15]); 

#define COMPUTE_F3S2_POS2(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[2], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[2], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[2], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[2], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[4], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[4], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[4], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[4], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[6], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[6], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[6], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[6], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[8], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[8], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[8], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[8], tmp_val.w, C[15]); 


template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_int8_nhwc_f3s1(
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
   int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = blockIdx.y;
    int c_stride = (paddingc >> 2);
    int c_idx = t_idx % c_stride;  // 4C 
    int hw_idx = t_idx / c_stride;
    int h_idx = hw_idx / out_width;
    int w_idx = hw_idx % out_width;

    int in_h = (h_idx * 4) * STRIRDE_H - pad_height; // 4H
    int in_w = w_idx * STRIRDE_W - pad_width;

    int64_t in_idx = n_idx * in_height * in_width * c_stride;
    in_idx += in_h * in_width * c_stride;
    in_idx += in_w * c_stride;
    in_idx += c_idx;

    int64_t out_idx = n_idx * out_height * out_width * c_stride;
    out_idx += (h_idx * 4) * out_width * c_stride;
    out_idx += w_idx * c_stride;
    out_idx += c_idx;

    int flt_off = c_idx;
    if (h_idx * 4 >= out_height) return;
    
    int C[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        C[i] = 0;
    }
    int pic_data[6], flt_data[3];
    DEFINE_POS(posX, posY, posZ, posW)

    int* int_filter = (int*)filter;
    int* int_input = (int*)input;
    int inh_stride = in_width * c_stride;
    int4 tmp_val;
    bool pred[6];
    pred[0] = (in_h+0>=0) && (in_h+0<in_height);
    pred[1] = (in_h+1>=0) && (in_h+1<in_height);
    pred[2] = (in_h+2>=0) && (in_h+2<in_height);
    pred[3] = (in_h+3>=0) && (in_h+3<in_height);
    pred[4] = (in_h+4>=0) && (in_h+4<in_height);
    pred[5] = (in_h+5>=0) && (in_h+5<in_height);
    if ((in_w >= 0) && (in_w < in_width)) {
        FETCH_FILTER(flt_data, int_filter, 0, 3, 6)
        FETCH_INPUT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F3S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F3S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F3S1_POS2(C, pic_data, tmp_val)
    }
    in_idx += c_stride;
//2
    if( (in_w+1>=0) && (in_w+1<in_width) ){
        FETCH_FILTER(flt_data, int_filter, 1, 4, 7)
        FETCH_INPUT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F3S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F3S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F3S1_POS2(C, pic_data, tmp_val)
    }
    in_idx += c_stride;

    //3
    if( (in_w+2>=0) && (in_w+2<in_width) ){
        FETCH_FILTER(flt_data, int_filter, 2, 5, 8)
        FETCH_INPUT(pic_data, int_input)
        
        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F3S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F3S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F3S1_POS2(C, pic_data, tmp_val)
    }

    int channel_idx = c_idx * 4;
    pred[0] = (channel_idx + 0 < channels);
    pred[1] = (channel_idx + 1 < channels);
    pred[2] = (channel_idx + 2 < channels);
    pred[3] = (channel_idx + 3 < channels);
    float4 regBias,regScale;
    float* fC = (float*)C;

    regScale.x = pred[0] ? flt_scale[channel_idx + 0] : 0.0f;
    regScale.y = pred[1] ? flt_scale[channel_idx + 1] : 0.0f;
    regScale.z = pred[2] ? flt_scale[channel_idx + 2] : 0.0f;
    regScale.w = pred[3] ? flt_scale[channel_idx + 3] : 0.0f;
    regScale.x *= pic_scale;
    regScale.y *= pic_scale;
    regScale.z *= pic_scale;
    regScale.w *= pic_scale;

    if (bias) {
        regBias.x  = pred[0] ? bias[channel_idx + 0] : 0.0f;
        regBias.y  = pred[1] ? bias[channel_idx + 1] : 0.0f;
        regBias.z  = pred[2] ? bias[channel_idx + 2] : 0.0f;
        regBias.w  = pred[3] ? bias[channel_idx + 3] : 0.0f;
    }
    int outData;
    int* dout = (int*)output;

    int64_t base_offset = n_idx * out_height * out_width * paddingc;
    base_offset += (h_idx * 4) * out_width * paddingc;
    base_offset += w_idx * paddingc;
    base_offset += c_idx * 4;

    if(h_idx*4+0<out_height) {
        fC[ 0] = C[ 0]*regScale.x;
        fC[ 1] = C[ 1]*regScale.y;
        fC[ 2] = C[ 2]*regScale.z;
        fC[ 3] = C[ 3]*regScale.w;
        if (bias) {
            fC[ 0] +=  regBias.x;
            fC[ 1] +=  regBias.y;
            fC[ 2] +=  regBias.z;
            fC[ 3] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT(fC, fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[0], C[1], C[2], C[3])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+1<out_height) {
        fC[ 4] = C[ 4]*regScale.x;
        fC[ 5] = C[ 5]*regScale.y;
        fC[ 6] = C[ 6]*regScale.z;
        fC[ 7] = C[ 7]*regScale.w;
        if (bias) {
            fC[ 4] +=  regBias.x;
            fC[ 5] +=  regBias.y;
            fC[ 6] +=  regBias.z;
            fC[ 7] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 4), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 4; i < 8; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[4], C[5], C[6], C[7])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+2<out_height) {
        fC[ 8] = C[ 8]*regScale.x;
        fC[ 9] = C[ 9]*regScale.y;
        fC[10] = C[ 10]*regScale.z;
        fC[11] = C[ 11]*regScale.w;
        if (bias) {
            fC[ 8] +=  regBias.x;
            fC[ 9] +=  regBias.y;
            fC[10] +=  regBias.z;
            fC[11] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 8), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 8; i < 12; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[8], C[9], C[10], C[11])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+3<out_height) {
        fC[12] = C[12]*regScale.x;
        fC[13] = C[13]*regScale.y;
        fC[14] = C[14]*regScale.z;
        fC[15] = C[15]*regScale.w;
        if (bias) {
            fC[12] +=  regBias.x;
            fC[13] +=  regBias.y;
            fC[14] +=  regBias.z;
            fC[15] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 12), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 12; i < 16; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[12], C[13], C[14], C[15])
        dout[out_idx] = outData;
    }
}

template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_int8_nhwc_f3s2(
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


    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = blockIdx.y;
    int c_stride = (paddingc >> 2);

    int c_idx = t_idx % c_stride;  // 4C 
    int hw_idx = t_idx /c_stride;
    int h_idx = hw_idx / out_width;
    int w_idx = hw_idx % out_width;

    int in_h = (h_idx * 4) * stride_height - pad_height; // 4H
    int in_w = w_idx * stride_width - pad_width;

    int64_t in_idx = n_idx * in_height * in_width * c_stride;
    in_idx += in_h * in_width * c_stride;
    in_idx += in_w * c_stride;
    in_idx += c_idx;

    int64_t out_idx = n_idx * out_height * out_width * c_stride;
    out_idx += (h_idx * 4) * out_width * c_stride;
    out_idx += w_idx * c_stride;
    out_idx += c_idx;

    int flt_off = c_idx;
    if (h_idx * 4 >= out_height) return;
    int C[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        C[i] = 0;
    }
    int pic_data[9], flt_data[3];
    DEFINE_POS(posX, posY, posZ, posW)
    int* int_filter = (int*)filter;
    int* int_input = (int*)input;
    int inh_stride = in_width * (paddingc >> 2);
    int4 tmp_val;
    bool pred[9];
    pred[0] = (in_h+0>=0) && (in_h+0<in_height);
    pred[1] = (in_h+1>=0) && (in_h+1<in_height);
    pred[2] = (in_h+2>=0) && (in_h+2<in_height);
    pred[3] = (in_h+3>=0) && (in_h+3<in_height);
    pred[4] = (in_h+4>=0) && (in_h+4<in_height);
    pred[5] = (in_h+5>=0) && (in_h+5<in_height);
    pred[6] = (in_h+6>=0) && (in_h+6<in_height);
    pred[7] = (in_h+7>=0) && (in_h+7<in_height);
    pred[8] = (in_h+8>=0) && (in_h+8<in_height);    
    if ((in_w >= 0) && (in_w < in_width)) {
        FETCH_FILTER(flt_data, int_filter, 0, 3, 6)
        FETCH_INPUT_NINE(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F3S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F3S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F3S2_POS2(C, pic_data, tmp_val)
    }
    in_idx += c_stride;
//2
    if( (in_w+1>=0) && (in_w+1<in_width) ){
        FETCH_FILTER(flt_data, int_filter, 1, 4, 7)
        FETCH_INPUT_NINE(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F3S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F3S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F3S2_POS2(C, pic_data, tmp_val)
    }
    in_idx += c_stride;

    //3
    if( (in_w+2>=0) && (in_w+2<in_width) ){
        FETCH_FILTER(flt_data, int_filter, 2, 5, 8)
        FETCH_INPUT_NINE(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F3S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F3S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F3S2_POS2(C, pic_data, tmp_val)
    }

    int channel_idx = c_idx * 4;
    pred[0] = (channel_idx + 0 < channels);
    pred[1] = (channel_idx + 1 < channels);
    pred[2] = (channel_idx + 2 < channels);
    pred[3] = (channel_idx + 3 < channels);
    float4 regBias,regScale;
    float* fC = (float*)C;
    regScale.x = pred[0] ? flt_scale[channel_idx + 0] : 0.0f;
    regScale.y = pred[1] ? flt_scale[channel_idx + 1] : 0.0f;
    regScale.z = pred[2] ? flt_scale[channel_idx + 2] : 0.0f;
    regScale.w = pred[3] ? flt_scale[channel_idx + 3] : 0.0f;
    regScale.x *= pic_scale;
    regScale.y *= pic_scale;
    regScale.z *= pic_scale;
    regScale.w *= pic_scale;
        if (bias) {
        regBias.x  = pred[0] ? bias[channel_idx + 0] : 0.0f;
        regBias.y  = pred[1] ? bias[channel_idx + 1] : 0.0f;
        regBias.z  = pred[2] ? bias[channel_idx + 2] : 0.0f;
        regBias.w  = pred[3] ? bias[channel_idx + 3] : 0.0f;
    }
    int outData;
    int* dout = (int*)output;

    int64_t base_offset = n_idx * out_height * out_width * paddingc;
    base_offset += (h_idx * 4) * out_width * paddingc;
    base_offset += w_idx * paddingc;
    base_offset += c_idx * 4;

    if(h_idx*4+0<out_height) {
        fC[ 0] = C[ 0]*regScale.x;
        fC[ 1] = C[ 1]*regScale.y;
        fC[ 2] = C[ 2]*regScale.z;
        fC[ 3] = C[ 3]*regScale.w;
        if (bias) {
            fC[ 0] +=  regBias.x;
            fC[ 1] +=  regBias.y;
            fC[ 2] +=  regBias.z;
            fC[ 3] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT(fC, fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[0], C[1], C[2], C[3])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+1<out_height) {
        fC[ 4] = C[ 4]*regScale.x;
        fC[ 5] = C[ 5]*regScale.y;
        fC[ 6] = C[ 6]*regScale.z;
        fC[ 7] = C[ 7]*regScale.w;
        if (bias) {
            fC[ 4] +=  regBias.x;
            fC[ 5] +=  regBias.y;
            fC[ 6] +=  regBias.z;
            fC[ 7] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 4), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 4; i < 8; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[4], C[5], C[6], C[7])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+2<out_height) {
        fC[ 8] = C[ 8]*regScale.x;
        fC[ 9] = C[ 9]*regScale.y;
        fC[10] = C[ 10]*regScale.z;
        fC[11] = C[ 11]*regScale.w;
        if (bias) {
            fC[ 8] +=  regBias.x;
            fC[ 9] +=  regBias.y;
            fC[10] +=  regBias.z;
            fC[11] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 8), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 8; i < 12; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[8], C[9], C[10], C[11])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+3<out_height) {
        fC[12] = C[12]*regScale.x;
        fC[13] = C[13]*regScale.y;
        fC[14] = C[14]*regScale.z;
        fC[15] = C[15]*regScale.w;
        if (bias) {
            fC[12] +=  regBias.x;
            fC[13] +=  regBias.y;
            fC[14] +=  regBias.z;
            fC[15] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 12), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 12; i < 16; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[12], C[13], C[14], C[15])
        dout[out_idx] = outData;
    }

}

#define COMPUTE_F5S1_POS0(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[0], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[0], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[0], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[0], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[1], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[1], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[1], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[1], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[2], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[2], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[2], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[2], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[3], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[3], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[3], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[3], tmp_val.w, C[15]); 

#define COMPUTE_F5S1_POS1(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[1], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[1], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[1], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[1], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[2], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[2], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[2], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[2], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[3], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[3], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[3], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[3], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[4], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[4], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[4], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[4], tmp_val.w, C[15]); 

#define COMPUTE_F5S1_POS2(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[2], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[2], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[2], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[2], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[3], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[3], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[3], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[3], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[4], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[4], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[4], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[4], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[5], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[5], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[5], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[5], tmp_val.w, C[15]); 

#define COMPUTE_F5S1_POS3(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[3], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[3], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[3], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[3], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[4], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[4], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[4], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[4], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[5], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[5], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[5], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[5], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[6], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[6], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[6], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[6], tmp_val.w, C[15]); 

#define COMPUTE_F5S1_POS4(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[4], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[4], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[4], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[4], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[5], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[5], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[5], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[5], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[6], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[6], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[6], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[6], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[7], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[7], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[7], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[7], tmp_val.w, C[15]); 


#define FETCH_INPUT_EIGHT(pic_data, int_input) \
    pic_data[0] = pred[0] ? int_input[in_idx + 0*inh_stride] : 0;\
    pic_data[1] = pred[1] ? int_input[in_idx + 1*inh_stride] : 0;\
    pic_data[2] = pred[2] ? int_input[in_idx + 2*inh_stride] : 0;\
    pic_data[3] = pred[3] ? int_input[in_idx + 3*inh_stride] : 0;\
    pic_data[4] = pred[4] ? int_input[in_idx + 4*inh_stride] : 0;\
    pic_data[5] = pred[5] ? int_input[in_idx + 5*inh_stride] : 0;\
    pic_data[6] = pred[6] ? int_input[in_idx + 6*inh_stride] : 0;\
    pic_data[7] = pred[7] ? int_input[in_idx + 7*inh_stride] : 0;

#define FETCH_INPUT_ELEVEN(pic_data, int_input) \
    pic_data[0] = pred[0] ? int_input[in_idx + 0*inh_stride] : 0;\
    pic_data[1] = pred[1] ? int_input[in_idx + 1*inh_stride] : 0;\
    pic_data[2] = pred[2] ? int_input[in_idx + 2*inh_stride] : 0;\
    pic_data[3] = pred[3] ? int_input[in_idx + 3*inh_stride] : 0;\
    pic_data[4] = pred[4] ? int_input[in_idx + 4*inh_stride] : 0;\
    pic_data[5] = pred[5] ? int_input[in_idx + 5*inh_stride] : 0;\
    pic_data[6] = pred[6] ? int_input[in_idx + 6*inh_stride] : 0;\
    pic_data[7] = pred[7] ? int_input[in_idx + 7*inh_stride] : 0;\
    pic_data[8] = pred[8] ? int_input[in_idx + 8*inh_stride] : 0;\
    pic_data[9] = pred[9] ? int_input[in_idx + 9*inh_stride] : 0;\
    pic_data[10] = pred[10] ? int_input[in_idx + 10*inh_stride] : 0;

#define FETCH_FILTER_FIVE(flt_data, int_filter, i, j, k, l, m) \
    flt_data[0] = int_filter[flt_off + i*c_stride]; \
    flt_data[1] = int_filter[flt_off + j*c_stride]; \
    flt_data[2] = int_filter[flt_off + k*c_stride]; \
    flt_data[3] = int_filter[flt_off + l*c_stride]; \
    flt_data[4] = int_filter[flt_off + m*c_stride]; 


#define COMPUTE_F5S2_POS0(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[0], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[0], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[0], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[0], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[2], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[2], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[2], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[2], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[4], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[4], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[4], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[4], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[6], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[6], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[6], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[6], tmp_val.w, C[15]); 

#define COMPUTE_F5S2_POS1(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[1], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[1], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[1], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[1], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[3], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[3], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[3], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[3], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[5], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[5], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[5], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[5], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[7], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[7], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[7], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[7], tmp_val.w, C[15]); 

#define COMPUTE_F5S2_POS2(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[2], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[2], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[2], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[2], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[4], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[4], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[4], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[4], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[6], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[6], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[6], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[6], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[8], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[8], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[8], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[8], tmp_val.w, C[15]); 

#define COMPUTE_F5S2_POS3(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[3], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[3], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[3], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[3], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[5], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[5], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[5], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[5], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[7], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[7], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[7], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[7], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[9], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[9], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[9], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[9], tmp_val.w, C[15]); 

#define COMPUTE_F5S2_POS4(C, pic_data, tmp_val) \
    C[ 0] = __dp4a(pic_data[4], tmp_val.x, C[ 0]); \
    C[ 1] = __dp4a(pic_data[4], tmp_val.y, C[ 1]); \
    C[ 2] = __dp4a(pic_data[4], tmp_val.z, C[ 2]); \
    C[ 3] = __dp4a(pic_data[4], tmp_val.w, C[ 3]); \
    C[ 4] = __dp4a(pic_data[6], tmp_val.x, C[ 4]); \
    C[ 5] = __dp4a(pic_data[6], tmp_val.y, C[ 5]); \
    C[ 6] = __dp4a(pic_data[6], tmp_val.z, C[ 6]); \
    C[ 7] = __dp4a(pic_data[6], tmp_val.w, C[ 7]); \
    C[ 8] = __dp4a(pic_data[8], tmp_val.x, C[ 8]); \
    C[ 9] = __dp4a(pic_data[8], tmp_val.y, C[ 9]); \
    C[10] = __dp4a(pic_data[8], tmp_val.z, C[10]); \
    C[11] = __dp4a(pic_data[8], tmp_val.w, C[11]); \
    C[12] = __dp4a(pic_data[10], tmp_val.x, C[12]); \
    C[13] = __dp4a(pic_data[10], tmp_val.y, C[13]); \
    C[14] = __dp4a(pic_data[10], tmp_val.z, C[14]); \
    C[15] = __dp4a(pic_data[10], tmp_val.w, C[15]);


template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_int8_nhwc_f5s1(
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
    int C[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        C[i] = 0;
    }

    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = blockIdx.y;
    int c_idx = t_idx % (paddingc >> 2);  // 4C 
    int hw_idx = t_idx / (paddingc >> 2);
    int c_stride = (paddingc >> 2);
    int h_idx = hw_idx / out_width;
    int w_idx = hw_idx % out_width;

    int in_h = (h_idx * 4) * stride_height - pad_height; // 4H
    int in_w = w_idx * stride_width - pad_width;

    int64_t in_idx = n_idx * in_height * in_width * (paddingc >> 2);
    in_idx += in_h * in_width * (paddingc >> 2);
    in_idx += in_w * (paddingc >> 2);
    in_idx += c_idx;

    int64_t out_idx = n_idx * out_height * out_width * (paddingc >> 2);
    out_idx += (h_idx * 4) * out_width * (paddingc >> 2);
    out_idx += w_idx * (paddingc >> 2);
    out_idx += c_idx;

    int flt_off = c_idx;
    if (h_idx * 4 >= out_height) return;

    int pic_data[8], flt_data[5];
    DEFINE_POS(posX, posY, posZ, posW)
    int* int_filter = (int*)filter;
    int* int_input = (int*)input;
    int inh_stride = in_width * (paddingc >> 2);
    int4 tmp_val;
    bool pred[8];
    pred[0] = (in_h+0>=0) && (in_h+0<in_height);
    pred[1] = (in_h+1>=0) && (in_h+1<in_height);
    pred[2] = (in_h+2>=0) && (in_h+2<in_height);
    pred[3] = (in_h+3>=0) && (in_h+3<in_height);
    pred[4] = (in_h+4>=0) && (in_h+4<in_height);
    pred[5] = (in_h+5>=0) && (in_h+5<in_height);
    pred[6] = (in_h+6>=0) && (in_h+6<in_height);
    pred[7] = (in_h+7>=0) && (in_h+7<in_height);
    if ((in_w >= 0) && (in_w < in_width)) {
        FETCH_FILTER_FIVE(flt_data, int_filter, 0, 5, 10, 15, 20)
        FETCH_INPUT_EIGHT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S1_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S1_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S1_POS4(C, pic_data, tmp_val)
    }
    in_idx += (paddingc >> 2);
//2
    if( (in_w+1>=0) && (in_w+1<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 1, 6, 11, 16, 21)
        FETCH_INPUT_EIGHT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S1_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S1_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S1_POS4(C, pic_data, tmp_val)
    }
    in_idx += (paddingc >> 2);

    //3
    if( (in_w+2>=0) && (in_w+2<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 2, 7, 12, 17, 22)
        FETCH_INPUT_EIGHT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S1_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S1_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S1_POS4(C, pic_data, tmp_val)
    }

    in_idx += (paddingc >> 2);

    //4
    if( (in_w+3>=0) && (in_w+3<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 3, 8, 13, 18, 23)
        FETCH_INPUT_EIGHT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S1_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S1_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S1_POS4(C, pic_data, tmp_val)
    }

    in_idx += (paddingc >> 2);

    //5
    if( (in_w+4>=0) && (in_w+4<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 4, 9, 14, 19, 24)
        FETCH_INPUT_EIGHT(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S1_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S1_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S1_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S1_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S1_POS4(C, pic_data, tmp_val)
    }

    int channel_idx = c_idx * 4;
    pred[0] = (channel_idx + 0 < channels);
    pred[1] = (channel_idx + 1 < channels);
    pred[2] = (channel_idx + 2 < channels);
    pred[3] = (channel_idx + 3 < channels);
    float4 regBias,regScale;
    float* fC = (float*)C;
    regScale.x = pred[0] ? flt_scale[channel_idx + 0] : 0.0f;
    regScale.y = pred[1] ? flt_scale[channel_idx + 1] : 0.0f;
    regScale.z = pred[2] ? flt_scale[channel_idx + 2] : 0.0f;
    regScale.w = pred[3] ? flt_scale[channel_idx + 3] : 0.0f;
    regScale.x *= pic_scale;
    regScale.y *= pic_scale;
    regScale.z *= pic_scale;
    regScale.w *= pic_scale;
        if (bias) {
        regBias.x  = pred[0] ? bias[channel_idx + 0] : 0.0f;
        regBias.y  = pred[1] ? bias[channel_idx + 1] : 0.0f;
        regBias.z  = pred[2] ? bias[channel_idx + 2] : 0.0f;
        regBias.w  = pred[3] ? bias[channel_idx + 3] : 0.0f;
    }
    int outData;
    int* dout = (int*)output;

    int64_t base_offset = n_idx * out_height * out_width * paddingc;
    base_offset += (h_idx * 4) * out_width * paddingc;
    base_offset += w_idx * paddingc;
    base_offset += c_idx * 4;

    if(h_idx*4+0<out_height) {
        fC[ 0] = C[ 0]*regScale.x;
        fC[ 1] = C[ 1]*regScale.y;
        fC[ 2] = C[ 2]*regScale.z;
        fC[ 3] = C[ 3]*regScale.w;
        if (bias) {
            fC[ 0] +=  regBias.x;
            fC[ 1] +=  regBias.y;
            fC[ 2] +=  regBias.z;
            fC[ 3] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT(fC, fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[0], C[1], C[2], C[3])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+1<out_height) {
        fC[ 4] = C[ 4]*regScale.x;
        fC[ 5] = C[ 5]*regScale.y;
        fC[ 6] = C[ 6]*regScale.z;
        fC[ 7] = C[ 7]*regScale.w;
        if (bias) {
            fC[ 4] +=  regBias.x;
            fC[ 5] +=  regBias.y;
            fC[ 6] +=  regBias.z;
            fC[ 7] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 4), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 4; i < 8; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[4], C[5], C[6], C[7])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+2<out_height) {
        fC[ 8] = C[ 8]*regScale.x;
        fC[ 9] = C[ 9]*regScale.y;
        fC[10] = C[ 10]*regScale.z;
        fC[11] = C[ 11]*regScale.w;
        if (bias) {
            fC[ 8] +=  regBias.x;
            fC[ 9] +=  regBias.y;
            fC[10] +=  regBias.z;
            fC[11] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 8), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 8; i < 12; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[8], C[9], C[10], C[11])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+3<out_height) {
        fC[12] = C[12]*regScale.x;
        fC[13] = C[13]*regScale.y;
        fC[14] = C[14]*regScale.z;
        fC[15] = C[15]*regScale.w;
        if (bias) {
            fC[12] +=  regBias.x;
            fC[13] +=  regBias.y;
            fC[14] +=  regBias.z;
            fC[15] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 12), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 12; i < 16; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[12], C[13], C[14], C[15])
        dout[out_idx] = outData;
    }
}


template<int TILE_H, int TILE_W, int SRC_TILE_H, int SRC_TILE_W, int KERNEL_H, int KERNEL_W, int STRIRDE_H, int STRIRDE_W>
__global__ void ppl_cuda_depthwise_int8_nhwc_f5s2(
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
    int C[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        C[i] = 0;
    }

    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = blockIdx.y;
    int c_idx = t_idx % (paddingc >> 2);  // 4C 
    int hw_idx = t_idx / (paddingc >> 2);
    int c_stride = (paddingc >> 2);
    int h_idx = hw_idx / out_width;
    int w_idx = hw_idx % out_width;

    int in_h = (h_idx * 4) * stride_height - pad_height; // 4H
    int in_w = w_idx * stride_width - pad_width;

    int64_t in_idx = n_idx * in_height * in_width * (paddingc >> 2);
    in_idx += in_h * in_width * (paddingc >> 2);
    in_idx += in_w * (paddingc >> 2);
    in_idx += c_idx;

    int64_t out_idx = n_idx * out_height * out_width * (paddingc >> 2);
    out_idx += (h_idx * 4) * out_width * (paddingc >> 2);
    out_idx += w_idx * (paddingc >> 2);
    out_idx += c_idx;

    int flt_off = c_idx;
    if (h_idx * 4 >= out_height) return;

    int pic_data[11], flt_data[5];
    DEFINE_POS(posX, posY, posZ, posW)
    int* int_filter = (int*)filter;
    int* int_input = (int*)input;
    int inh_stride = in_width * (paddingc >> 2);
    int4 tmp_val;
    bool pred[11];
    pred[0] = (in_h+0>=0) && (in_h+0<in_height);
    pred[1] = (in_h+1>=0) && (in_h+1<in_height);
    pred[2] = (in_h+2>=0) && (in_h+2<in_height);
    pred[3] = (in_h+3>=0) && (in_h+3<in_height);
    pred[4] = (in_h+4>=0) && (in_h+4<in_height);
    pred[5] = (in_h+5>=0) && (in_h+5<in_height);
    pred[6] = (in_h+6>=0) && (in_h+6<in_height);
    pred[7] = (in_h+7>=0) && (in_h+7<in_height);
    pred[8] = (in_h+8>=0) && (in_h+8<in_height);
    pred[9] = (in_h+9>=0) && (in_h+9<in_height);
    pred[10] = (in_h+10>=0) && (in_h+10<in_height);

    if ((in_w >= 0) && (in_w < in_width)) {
        FETCH_FILTER_FIVE(flt_data, int_filter, 0, 5, 10, 15, 20)
        FETCH_INPUT_ELEVEN(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S2_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S2_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S2_POS4(C, pic_data, tmp_val)
    }
    in_idx += (paddingc >> 2);
//2
    if( (in_w+1>=0) && (in_w+1<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 1, 6, 11, 16, 21)
        FETCH_INPUT_ELEVEN(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S2_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S2_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S2_POS4(C, pic_data, tmp_val)
    }
    in_idx += (paddingc >> 2);

    //3
    if( (in_w+2>=0) && (in_w+2<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 2, 7, 12, 17, 22)
        FETCH_INPUT_ELEVEN(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S2_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S2_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S2_POS4(C, pic_data, tmp_val)
    }

    in_idx += (paddingc >> 2);

    //4
    if( (in_w+3>=0) && (in_w+3<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 3, 8, 13, 18, 23)
        FETCH_INPUT_ELEVEN(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S2_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S2_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S2_POS4(C, pic_data, tmp_val)
    }

    in_idx += (paddingc >> 2);

    //5
    if( (in_w+4>=0) && (in_w+4<in_width) ){
        FETCH_FILTER_FIVE(flt_data, int_filter, 4, 9, 14, 19, 24)
        FETCH_INPUT_ELEVEN(pic_data, int_input)

        GET_POS4_VAL(tmp_val, flt_data[0])
        COMPUTE_F5S2_POS0(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[1])
        COMPUTE_F5S2_POS1(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[2])
        COMPUTE_F5S2_POS2(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[3])
        COMPUTE_F5S2_POS3(C, pic_data, tmp_val)

        GET_POS4_VAL(tmp_val, flt_data[4])
        COMPUTE_F5S2_POS4(C, pic_data, tmp_val)
    }

    int channel_idx = c_idx * 4;
    pred[0] = (channel_idx + 0 < channels);
    pred[1] = (channel_idx + 1 < channels);
    pred[2] = (channel_idx + 2 < channels);
    pred[3] = (channel_idx + 3 < channels);
    float4 regBias,regScale;
    float* fC = (float*)C;
    regScale.x = pred[0] ? flt_scale[channel_idx + 0] : 0.0f;
    regScale.y = pred[1] ? flt_scale[channel_idx + 1] : 0.0f;
    regScale.z = pred[2] ? flt_scale[channel_idx + 2] : 0.0f;
    regScale.w = pred[3] ? flt_scale[channel_idx + 3] : 0.0f;
    regScale.x *= pic_scale;
    regScale.y *= pic_scale;
    regScale.z *= pic_scale;
    regScale.w *= pic_scale;
    if (bias) {
        regBias.x  = pred[0] ? bias[channel_idx + 0] : 0.0f;
        regBias.y  = pred[1] ? bias[channel_idx + 1] : 0.0f;
        regBias.z  = pred[2] ? bias[channel_idx + 2] : 0.0f;
        regBias.w  = pred[3] ? bias[channel_idx + 3] : 0.0f;
    }
    int outData;
    int* dout = (int*)output;

    int64_t base_offset = n_idx * out_height * out_width * paddingc;
    base_offset += (h_idx * 4) * out_width * paddingc;
    base_offset += w_idx * paddingc;
    base_offset += c_idx * 4;

    if(h_idx*4+0<out_height) {
        fC[ 0] = C[ 0]*regScale.x;
        fC[ 1] = C[ 1]*regScale.y;
        fC[ 2] = C[ 2]*regScale.z;
        fC[ 3] = C[ 3]*regScale.w;
        if (bias) {
            fC[ 0] +=  regBias.x;
            fC[ 1] +=  regBias.y;
            fC[ 2] +=  regBias.z;
            fC[ 3] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT(fC, fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[0], C[1], C[2], C[3])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+1<out_height) {
        fC[ 4] = C[ 4]*regScale.x;
        fC[ 5] = C[ 5]*regScale.y;
        fC[ 6] = C[ 6]*regScale.z;
        fC[ 7] = C[ 7]*regScale.w;
        if (bias) {
            fC[ 4] +=  regBias.x;
            fC[ 5] +=  regBias.y;
            fC[ 6] +=  regBias.z;
            fC[ 7] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 4), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 4; i < 8; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[4], C[5], C[6], C[7])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+2<out_height) {
        fC[ 8] = C[ 8]*regScale.x;
        fC[ 9] = C[ 9]*regScale.y;
        fC[10] = C[ 10]*regScale.z;
        fC[11] = C[ 11]*regScale.w;
        if (bias) {
            fC[ 8] +=  regBias.x;
            fC[ 9] +=  regBias.y;
            fC[10] +=  regBias.z;
            fC[11] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 8), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 8; i < 12; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[8], C[9], C[10], C[11])
        dout[out_idx] = outData;
    }
    out_idx += out_width * c_stride;
    base_offset += out_width + paddingc;

    if(h_idx*4+3<out_height) {
        fC[12] = C[12]*regScale.x;
        fC[13] = C[13]*regScale.y;
        fC[14] = C[14]*regScale.z;
        fC[15] = C[15]*regScale.w;
        if (bias) {
            fC[12] +=  regBias.x;
            fC[13] +=  regBias.y;
            fC[14] +=  regBias.z;
            fC[15] +=  regBias.w;        
        }
        FUSE_PROCESS_FLOAT((fC + 12), fuse_params, 4, 1, base_offset)
        #pragma unroll
        for(int i = 12; i < 16; i++)
        {
            C[i] = __float2int_rn(fC[i] * out_scale);
            C[i] = C[i] > 127 ? 127 : C[i] < -128 ? -128 : C[i];
        }
        PACK4INT(outData, C[12], C[13], C[14], C[15])
        dout[out_idx] = outData;
    }

}

#endif
