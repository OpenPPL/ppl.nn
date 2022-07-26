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

#define _4HALF2_        4
#define _4INT_          4
#define _8HALF_         8
#define _INT4_TO_4INT_  4
#define _INT4_TO_8HALF_ 8
#define _INT4_TO_4FLOAT_  4
#define MAX_SPLIT_SIZE  18

#include <cuda_fp16.h>
#include "cudakernel/common/macro.h"

__global__ void MergeConvSplitResults(
    int4 *input,
    int4 *output,
    int split_height_v1,
    int split_width_v8,
    int out_hw,
    int split,
    int has_bias,
    const int4 *bias,
    int has_relu,
    const __half2 clip_min,
    bool has_clip,
    const __half2 clip_max,
    int has_prelu,
    const void *prelu,
    bool has_elt,
    const int4 *pre_data,
    int has_elt_relu,
    const __half2 elt_clip_min,
    bool has_elt_clip,
    const __half2 elt_clip_max,
    int has_elt_prelu,
    const void *elt_prelu,
    const __half leaky,
    const __half elt_leaky,
    bool has_concat,
    int concat_offset_v8,
    int concat_stride_v8)
{
#if (__CUDA_ARCH__ >= 600) && (__CUDACC_VER_MAJOR__ >= 9)
    int k_id       = blockIdx.y * blockDim.x + threadIdx.x;
    int64_t nhw_id = blockIdx.x;

    int off = nhw_id * split_width_v8 + k_id;

    const int4 ZEROv4 = {0, 0, 0, 0};
    bool is_in_range  = k_id < split_width_v8;

    int4 merge_v4, bias_v4;
    int4 split_v4[MAX_SPLIT_SIZE];

    __half2 *h2_merge = (__half2 *)&merge_v4;
    __half2 *h2_split = (__half2 *)&split_v4;
    __half2 *h2_bias  = (__half2 *)&bias_v4;

    merge_v4 = is_in_range ? input[off] : ZEROv4;

#pragma unroll
    for (int i = 1; i < split; i++) {
        split_v4[i] = is_in_range ? input[off + i * split_height_v1 * split_width_v8] : ZEROv4;
    }

#pragma unroll
    for (int i = 1; i < split; i++) {
        for (int j = 0; j < _4HALF2_; j++)
            h2_merge[j] = __hadd2(h2_merge[j], h2_split[i * _4HALF2_ + j]);
    }

    if (has_bias) {
        bias_v4 = is_in_range ? ((int4 *)bias)[k_id] : ZEROv4;

#pragma unroll
        for (int j = 0; j < _4HALF2_; j++)
            h2_merge[j] = __hadd2(h2_merge[j], h2_bias[j]);
    }

    int *merge_v1   = (int *)&merge_v4;
    __half *h_merge = (__half *)&merge_v4;

    if (has_relu) {
        if (has_relu == 1) {
#pragma unroll
            for (int i = 0; i < _4HALF2_; i++)
                merge_v1[i] = __vmaxs2(merge_v1[i], 0);
        } else {
            __half *h_merge = (__half*)merge_v1;
#pragma unroll
            for (int i = 0; i < _INT4_TO_8HALF_; i++) {
                h_merge[i] = __expf((float)h_merge[i]) / (1.f + __expf((float)h_merge[i]));
            }
        }
    } else if (has_clip) {
#pragma unroll
        for (int i = 0; i < _4HALF2_; i++) {
            h2_merge[i].x = __hgt(h2_merge[i].x, clip_max.x) ? clip_max.x : h2_merge[i].x;
            h2_merge[i].y = __hgt(h2_merge[i].y, clip_max.x) ? clip_max.x : h2_merge[i].y;
            h2_merge[i].x = __hlt(h2_merge[i].x, clip_min.x) ? clip_min.x : h2_merge[i].x;
            h2_merge[i].y = __hlt(h2_merge[i].y, clip_min.x) ? clip_min.x : h2_merge[i].y;
        }
    } else if (has_prelu) {
        if (has_prelu == 1) {
#pragma unroll
            for (int i = 0; i < _INT4_TO_8HALF_; i++)
                if (__hlt(h_merge[i], 0))
                    h_merge[i] = __hmul(h_merge[i], leaky);
        }
        if (has_prelu == 2) {
            int4 scale_v4   = ((int4 *)prelu)[k_id];
            __half *h_scale = (__half *)&scale_v4;

#pragma unroll
            for (int i = 0; i < _INT4_TO_8HALF_; i++)
                if (__hlt(h_merge[i], 0))
                    h_merge[i] = __hmul(h_merge[i], h_scale[i]);
        }
        if (has_prelu == 3) {
            int4 elt_v4   = ((int4 *)prelu)[off];
            __half *h_elt = (__half *)&elt_v4;

#pragma unroll
            for (int i = 0; i < _INT4_TO_8HALF_; i++)
                if (__hlt(h_merge[i], 0))
                    h_merge[i] = __hmul(h_merge[i], h_elt[i]);
        }
    }

    if (has_elt) {
        int4 eltV4     = is_in_range ? pre_data[off] : ZEROv4;
        __half2 *h2Elt = (__half2 *)&eltV4;

        for (int i = 0; i < _INT4_TO_4INT_; i++)
            h2_merge[i] = __hadd2(h2_merge[i], h2Elt[i]);
    }

    if (has_elt_relu) {
        if (has_elt_relu == 1) {
            for (int i = 0; i < _4HALF2_; i++)
                merge_v1[i] = __vmaxs2(merge_v1[i], 0);
        } else {
            __half *h_merge = (__half*)merge_v1;
#pragma unroll
            for (int i = 0; i < _INT4_TO_8HALF_; i++) {
                h_merge[i] = __expf((float)h_merge[i]) / (1.f + __expf((float)h_merge[i]));
            }
        }
    } else if (has_elt_clip) {
        for (int i = 0; i < _4HALF2_; i++) {
            h2_merge[i].x = __hgt(h2_merge[i].x, elt_clip_max.x) ? elt_clip_max.x : h2_merge[i].x;
            h2_merge[i].y = __hgt(h2_merge[i].y, elt_clip_max.x) ? elt_clip_max.x : h2_merge[i].y;
            h2_merge[i].x = __hlt(h2_merge[i].x, elt_clip_min.x) ? elt_clip_min.x : h2_merge[i].x;
            h2_merge[i].y = __hlt(h2_merge[i].y, elt_clip_min.x) ? elt_clip_min.x : h2_merge[i].y;
        }
    } else if (has_elt_prelu) {
        if (has_prelu == 1) {
            for (int i = 0; i < _INT4_TO_8HALF_; i++)
                if (__hlt(h_merge[i], 0))
                    h_merge[i] = __hmul(h_merge[i], elt_leaky);
        }

        if (has_elt_prelu == 2) {
            int4 scale_v4   = ((int4 *)prelu)[k_id];
            __half *h_scale = (__half *)&scale_v4;

            for (int i = 0; i < _INT4_TO_8HALF_; i++)
                if (__hlt(h_merge[i], 0))
                    h_merge[i] = __hmul(h_merge[i], h_scale[i]);
        }

        if (has_elt_prelu == 3) {
            int4 elt_v4   = ((int4 *)prelu)[off];
            __half *h_elt = (__half *)&elt_v4;

            for (int i = 0; i < _INT4_TO_8HALF_; i++)
                if (__hlt(h_merge[i], 0))
                    h_merge[i] = __hmul(h_merge[i], h_elt[i]);
        }
    }

    int concat_v8_off = 0;
    if (has_concat) {
        concat_v8_off = concat_offset_v8 + nhw_id * concat_stride_v8;
        off           = concat_v8_off + k_id;
    }

    if (is_in_range)
        output[off] = merge_v4;

#endif
}


__global__ void MergeConvSplitResultsFp32(
        int4* input,             int* output, 
	    int split_height_v1,     int split_width_v8, 
	    int out_hw,              int split, 
        int has_bias,            const int4* bias,
        int has_relu,            const float clip_min,
	    bool has_clip,           const float clip_max,
        int has_prelu,           const void* prelu,
        bool has_elt,            const int4* pre_data,
        int has_elt_relu,        const float elt_clip_min,
	    bool has_elt_clip,       const float elt_clip_max,
        int has_elt_prelu,       const void* elt_prelu,
        const float leaky,       const float elt_leaky,
        bool has_concat,         int concat_offset_v8,
        int concat_stride_v8,
        float out_scale,         float pre_scale)
{
#if (__CUDA_ARCH__ >= 600) && (__CUDACC_VER_MAJOR__ >= 9)
    int k_id = blockIdx.y * blockDim.x + threadIdx.x;
    int64_t nhw_id = blockIdx.x;

    int off  = nhw_id * split_width_v8 + k_id;

    const int4 ZEROv4 = {0, 0, 0, 0};
    bool is_in_range = k_id < split_width_v8;

    int4 merge_v4, bias_v4;
    int4 split_v4[MAX_SPLIT_SIZE];

    float * f_merge = (float *) &merge_v4;
    float * f_split = (float *) &split_v4;
    float * f_bias  = (float *) &bias_v4;

    merge_v4 = is_in_range ? input[off] : ZEROv4;

#pragma unroll
    for(int i = 1; i < split; i++) {
        split_v4[i] = is_in_range ? input[off + i * split_height_v1 * split_width_v8] : ZEROv4;
    }

#pragma unroll
    for(int i = 1; i < split; i++) {
	    for(int j = 0; j < _INT4_TO_4FLOAT_; j++)
	        f_merge[j] = f_merge[j] + f_split[i * _4INT_ + j];
    }

    if(has_bias)
    {
        bias_v4 = is_in_range ? ((int4 *) bias) [k_id] : ZEROv4;

#pragma unroll
	    for(int j = 0; j < _INT4_TO_4FLOAT_; j++)
	        f_merge[j] = f_merge[j] + f_bias[j];
    }

    int *    merge_v1  = (int *)    &merge_v4;
    //float * h_merge = (float *) &merge_v4;

    if(has_relu)
    {
        if(has_relu == 1){
            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
                merge_v1[i] = merge_v1[i] >= 0? merge_v1[i] : 0.f;
	    } else {

            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
	            f_merge[i]  = __expf(f_merge[i]) / (1.f + __expf(f_merge[i]));
	    }
    }
    else if(has_clip) {
#pragma unroll
        for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        {
	        f_merge[i] = (f_merge[i] > clip_max) ? clip_max : f_merge[i];
	        f_merge[i] = (f_merge[i] < clip_min) ? clip_min : f_merge[i];
        }
    }
    else if(has_prelu) {
        if(has_prelu == 1)
        {
#pragma unroll
            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        	    if(f_merge[i] < 0)
                    f_merge[i] = f_merge[i] * leaky;
        }
        if(has_prelu == 2)
        {
            int4     scale_v4 = ( (int4 *) prelu) [k_id];
            float * h_scale  = (float *) &scale_v4;

#pragma unroll
            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        	    if(f_merge[i] < 0)
                    f_merge[i] = f_merge[i] * h_scale[i];
        }
        if(has_prelu == 3)
        {
            int4   elt_v4 = ( (int4 *) prelu) [off];
            float* h_elt = (float *) &elt_v4;

#pragma unroll
            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        	    if(f_merge[i] < 0)
                    f_merge[i] = f_merge[i] * h_elt[i];
        }
    }

    if(has_elt) {
	    int  elt_v4 = is_in_range ? ((int *)pre_data)[off] : 0;
	    int8_t *elt = (int8_t *) &elt_v4;

#pragma unroll
	    for(int i = 0; i < _INT4_TO_4INT_; i++)
			f_merge[i] += (int)elt[i] * pre_scale;
    }

    if(has_elt_relu) {
        if(has_elt_relu == 1) {
            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
                merge_v1[i] = (merge_v1[i] >= 0)? merge_v1[i] : 0;
	    } else{

            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
	            f_merge[i]  = __expf(f_merge[i]) / (1.f + __expf(f_merge[i]));
	        
	    }
    } else if(has_elt_clip) {
        for(int i = 0; i < _INT4_TO_4FLOAT_; i++) {
	        f_merge[i] = (f_merge[i] > elt_clip_max) ? elt_clip_max : f_merge[i];
	        f_merge[i] = (f_merge[i] < elt_clip_min) ? elt_clip_min : f_merge[i];
        }
    }
    else if(has_elt_prelu) {
        if(has_prelu == 1) {
            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        	    if((f_merge[i] < 0))
                    f_merge[i] = (f_merge[i] * elt_leaky);
        }

        if(has_elt_prelu == 2) {
            int4    scale_v4 = ((int4  *) prelu) [k_id];
            float* h_scale  = (float *) &scale_v4;

            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        	    if((f_merge[i] < 0))
                    f_merge[i] = (f_merge[i] * h_scale[i]);
        }

        if(has_elt_prelu == 3) {
            int4    elt_v4 = ((int4 *)prelu) [off];
            float* h_elt  = (float *) &elt_v4;

            for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
        	    if((f_merge[i] < 0))
                    f_merge[i] = (f_merge[i] * h_elt[i]);
        }
    }
#define packchar4(_outData, x, y, z, w){ \
    if (x>127)	x = 127;                 \
    if (x<-128) x = -128;                \
    if (y>127)	y = 127;                 \
    if (y<-128) y = -128;                \
    if (z>127)	z = 127;                 \
    if (z<-128) z = -128;                \
    if (w>127)	w = 127;                 \
    if (w<-128) w = -128;                \
    x = (0xffu & (int8_t)x);             \
    y = (0xffu & (int8_t)y) << 8;        \
    z = (0xffu & (int8_t)z) << 16;       \
    w = (0xffu & (int8_t)w) << 24;         \
    _outData = w | z | y | x;/*(x,y,z,w)*/\
}
    for(int i = 0; i < _INT4_TO_4FLOAT_; i++)
	    merge_v1[i] = __float2int_rn(f_merge[i]*out_scale);
    int outData;
    packchar4(outData, merge_v1[0], merge_v1[1], merge_v1[2], merge_v1[3]);
#undef packchar4

    int concat_v8_off = 0;
    if(has_concat){
	    concat_v8_off = concat_offset_v8 + nhw_id * concat_stride_v8;
	    off = concat_v8_off + k_id;
    }
    
    if(is_in_range) output[off] = outData;

#endif
}
