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

#include <vector>
#include <cuda.h>
#include <assert.h>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <nvrtc.h>
#include <algorithm>

#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/nn/conv/gene_kernel.h"
#include "cudakernel/common/cuda_check.h"
#include "kernel_type.h"
#include "conv_common.h"
#include "common/init_lut.h"
#include "common/merge_split.h"

#include "ppl/nn/engines/cuda/module/cuda_compiler.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"

#include "float.h"

#define TIMES 4
#define __INT4__ 4

#define INT8_SPK_KPARAM_LIST                               \
        pad_input,                                         \
        d_flt,                                             \
        conv_out,                                          \
        kloop_num,                                         \
        in_lut,                   in_lut_size,             \
        flt_lut,                  flt_lut_size,            \
        num_chl_per_spk_head,                              \
        num_chl_per_spk_tail,                              \
        in_hw, out_hw,                                     \
        flt_hw, splitk,                                    \
        conv_param.in_height,     conv_param.in_width,     \
        conv_param.in_num,        conv_param.num_grp,      \
        num_chl_per_grp,          num_chl_per_grp_pad,     \
        conv_param.flt_height,    conv_param.flt_width,    \
        num_flt_per_grp,          num_flt_per_grp_pad,     \
        conv_param.out_height,    conv_param.out_width,    \
        conv_param.stride_height, conv_param.stride_width, \
        conv_param.pad_height,    conv_param.pad_width,    \
        conv_param.hole_height,   conv_param.hole_width,   \
        conv_param.has_bias,      (int *)bias,             \
        quant_param.in_scale,     quant_param.d_flt_scale

#define INT8_LUT_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                                                                  \
            in_lut,                        in_lut_size,                                 \
    	    flt_lut,                       flt_lut_size,                                \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            quant_param.in_scale,          quant_param.d_flt_scale,                     \
            quant_param.out_scale,         quant_param.pre_scale,                       \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_prelu,          (const void *) fuse_param.prelu,             \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,         \
            leaky,                         elt_leaky,                                   \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8


#define INT8_IDX_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                      koff_num_pad,                               \
            in_hw,                         out_hw,                                      \
            flt_hw,                        out_nhw,                                     \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            conv_param.num_chl,            num_chl_per_grp,                             \
            in_chl_per_grp_pad,            flt_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            quant_param.in_scale,          quant_param.d_flt_scale,                     \
            quant_param.out_scale,         quant_param.pre_scale,                       \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_prelu,          (const void *) fuse_param.prelu,             \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,         \
            leaky,                         elt_leaky,                                   \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

#define INT8_MERGE_KPARAM_LIST \
        	conv_out,                      (int*)final_out,                             \
        	spk_height_v1,                 spk_width_v4,                                \
        	out_hw,                        splitk * splitf,                             \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_prelu,          (const void *) fuse_param.prelu,             \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,         \
            leaky,                         elt_leaky,                                   \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8,                                                           \
            quant_param.out_scale,         quant_param.pre_scale                        \

static std::vector<kernel_info_t> g_int8_kernel_container;
static bool is_g_int8_kernel_container_initialized = false;

static std::unordered_map<size_t, algo_param_t> g_conv_shape_hash;

__inline__ void InitializeKernelContainer(std::vector<kernel_info_t> &g_kernel_container, ppl::common::datatype_t type)
{
    if (type == ppl::common::DATATYPE_INT8) {
#ifndef PPLNN_ENABLE_CUDA_JIT
        InitializeInt82spkConvF1KernelContainer(g_int8_kernel_container);
        InitializeInt82spkConvF3KernelContainer(g_int8_kernel_container);
        InitializeInt82spkConvFNKernelContainer(g_int8_kernel_container);
        InitializeInt82spkConvFSKernelContainer(g_int8_kernel_container);

        InitializeInt8IdxnConvKernelContainer(g_int8_kernel_container);
#endif
    }
    is_g_int8_kernel_container_initialized = true;
}

__inline__ size_t GetConvShapeHashKey(conv_param_t &conv_param)
{
    return std::hash<std::string>{}(GetConvShapeString(conv_param));
}

/* -----------------  INT8 KERNEL ------------------ */

double PPLCUDAConvolutionSelectKernelInt8(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
        int4* bias,
        int4* d_temp_buf, 
        algo_param_t & algo_param,
        conv_param_t &conv_param, 
        quant_param_t &quant_param,
        fuse_param_t &fuse_param,
	    uint64_t workspace)
{
    if(!is_g_int8_kernel_container_initialized)
        InitializeKernelContainer(g_int8_kernel_container, type);

    size_t conv_shape_hash = GetConvShapeHashKey(conv_param);

    std::unordered_map<size_t, algo_param_t>::const_iterator conv_shape_hash_iterator = g_conv_shape_hash.find(conv_shape_hash);

    if(conv_shape_hash_iterator != g_conv_shape_hash.end()) {
        algo_param.kid    = conv_shape_hash_iterator->second.kid;
        algo_param.splitk = conv_shape_hash_iterator->second.splitk;
        algo_param.splitf = conv_shape_hash_iterator->second.splitf;

        return 0.0f;
    }

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input = d_input;
    int4 *pad_output = d_output;

    if(is_in_grp_pad) {
	    pad_input = d_temp_buf; 
	    buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if(is_out_grp_pad) {
	    pad_output = d_temp_buf + buf_off_v4;
	    buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    } 

    int4 * final_out = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;

    float clip_min     = fuse_param.clip_min;
    float clip_max     = fuse_param.clip_max;
    float elt_clip_min = fuse_param.elt_clip_min;
    float elt_clip_max = fuse_param.elt_clip_max;
    float leaky        = fuse_param.leaky;
    float elt_leaky    = fuse_param.elt_leaky;

    float minTime = FLT_MAX;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    const int SPLITK_OPTIONS[] = {1, 2, 4, 8};

    for(unsigned int spk = 0; spk < 4; spk++) {
        unsigned int splitk = SPLITK_OPTIONS[spk];

        for(unsigned int kid = 0; kid < g_int8_kernel_container.size(); kid++) {

            unsigned int splitf = (g_int8_kernel_container[kid].ktype == CONV_2SPK_FS) ? flt_hw : 1;
            if(!g_int8_kernel_container[kid].CheckKernelTypeFeasibleInt8(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk)) continue;

            if(!g_int8_kernel_container[kid].CheckSplitkFeasible(num_chl_per_grp, splitk)) continue;

            if(!g_int8_kernel_container[kid].CheckSplitfFeasible(splitf, splitk)) continue;


            int4 *conv_out = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

            dim3 block_size, grid_size;

            block_size.x = g_int8_kernel_container[kid].cta_size_in_thd;
            block_size.y = 1;
            block_size.z = 1;

            grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_int8_kernel_container[kid].tile_m_per_cta);
            grid_size.y = DivUp(num_flt_per_grp_pad, g_int8_kernel_container[kid].tile_n_per_cta);
            grid_size.z = conv_param.num_grp * splitk * splitf;

	        cudaEventRecord(begin, stream);

	        for(int i = 0; i < TIMES; i++) {
                if(g_int8_kernel_container[kid].ktype == CONV_IDXN_C2 || g_int8_kernel_container[kid].ktype == CONV_IDXN_C4 || \
                        g_int8_kernel_container[kid].ktype == CONV_IDXN_C32) {
                    int tile_k_per_step = g_int8_kernel_container[kid].tile_k_per_step;

                    int img_pad_size    = pad_size;
                    int flt_pad_size    = g_int8_kernel_container[kid].flt_pad_size;
                    int out_nhw         = out_hw * conv_param.in_num;

                    int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
                    int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
                    int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	                int kloop_num        = DivUp(flt_hw * flt_chl_per_grp_pad, g_int8_kernel_container[kid].tile_k_per_cta);
                    int koff_num_pad      = Align(kloop_num * (g_int8_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

                    (g_int8_kernel_container[kid].int8_idx_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_IDX_KPARAM_LIST);
                }
                else if(g_int8_kernel_container[kid].ktype == CONV_2SPK_F1 || g_int8_kernel_container[kid].ktype == CONV_2SPK_F3 || \
                        g_int8_kernel_container[kid].ktype == CONV_2SPK_FN || g_int8_kernel_container[kid].ktype == CONV_2SPK_FS) {

	                int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_int8_kernel_container[kid].tile_k_per_cta);

                    lut_t in_lut, flt_lut;
                    int in_lut_size, flt_lut_size;
                
                    InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                            conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                            num_chl_per_grp_pad, conv_param.num_grp, g_int8_kernel_container[kid].tile_k_per_cta, pad_size);

                    InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                            g_int8_kernel_container[kid].tile_k_per_cta, pad_size);

                    if(splitk == 1) {
                        (g_int8_kernel_container[kid].int8_lut_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_LUT_KPARAM_LIST);

                    }
		        else {
                        int num_chl_per_spk_head, num_chl_per_spk_tail;
                        InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_int8_kernel_container[kid].tile_k_per_cta, splitk);
    
                        (g_int8_kernel_container[kid].int8_spk_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_SPK_KPARAM_LIST);
                    }

                    if(splitk > 1 || splitf > 1) {
                        int spk_width_v4   = num_flt_per_grp_pad * conv_param.num_grp / __INT4__;
                        int spk_height_v1  = out_hw * conv_param.in_num;

                        dim3 merge_grid_size, merge_block_size;
                        merge_block_size.x = 64; // empirical value
                        merge_block_size.y = 1;
                        merge_block_size.z = 1;

                        merge_grid_size.x  = spk_height_v1;
                        merge_grid_size.y  = DivUp(spk_width_v4, merge_block_size.x);
                        merge_grid_size.z  = 1;

                        MergeConvSplitResultsFp32<<<merge_grid_size, merge_block_size, 0, stream>>>(INT8_MERGE_KPARAM_LIST);
                    }
                }
            }

	        cudaEventRecord(end, stream);
	        cudaEventSynchronize(end);
	        cudaEventElapsedTime(&elapsed, begin, end);
	        if(elapsed < minTime){
                algo_param.kid = kid;
                algo_param.splitk = splitk;
                algo_param.splitf = splitf;
	            minTime = elapsed;
            }
        }

    }

    if(is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    g_conv_shape_hash[conv_shape_hash] = algo_param;

    return minTime;
}

void PPLCUDAConvolutionForwardImpInt8(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
        int4* bias,
        int4* d_temp_buf,
        algo_param_t& algo_param,
        conv_param_t &conv_param,
        quant_param_t &quant_param,
        fuse_param_t &fuse_param)
{
    if(!is_g_int8_kernel_container_initialized)
        InitializeKernelContainer(g_int8_kernel_container, type);

    unsigned int kid = algo_param.kid;
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    //printf("kernel name: %d, %d, %s\n", splitk, splitf, g_int8_kernel_container[kid].kname.c_str());
    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input = d_input;
    int4 *pad_output = d_output;

    if(is_in_grp_pad) {
	    pad_input = d_temp_buf; 
	    buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);

    }

    if(is_out_grp_pad) {
	    pad_output = d_temp_buf + buf_off_v4;
	    buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    } 

    int4 *final_out  = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    float clip_min     = fuse_param.clip_min;
    float clip_max     = fuse_param.clip_max;
    float elt_clip_min = fuse_param.elt_clip_min;
    float elt_clip_max = fuse_param.elt_clip_max;
    float leaky        = fuse_param.leaky;
    float elt_leaky    = fuse_param.elt_leaky;

    dim3 block_size, grid_size;

    block_size.x = g_int8_kernel_container[kid].cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    grid_size.x  = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_int8_kernel_container[kid].tile_m_per_cta);
    grid_size.y  = DivUp(num_flt_per_grp_pad, g_int8_kernel_container[kid].tile_n_per_cta);
    grid_size.z  = conv_param.num_grp * splitk * splitf;

    if(g_int8_kernel_container[kid].ktype == CONV_IDXN_C2 || g_int8_kernel_container[kid].ktype == CONV_IDXN_C4 || \
            g_int8_kernel_container[kid].ktype == CONV_IDXN_C32) {
        int img_pad_size = pad_size;
        int flt_pad_size = g_int8_kernel_container[kid].flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	    int kloop_num = DivUp(flt_hw * flt_chl_per_grp_pad, g_int8_kernel_container[kid].tile_k_per_cta);
        int koff_num_pad = Align(kloop_num * (g_int8_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

        (g_int8_kernel_container[kid].int8_idx_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_IDX_KPARAM_LIST);

    } else if(g_int8_kernel_container[kid].ktype == CONV_2SPK_F1 || g_int8_kernel_container[kid].ktype == CONV_2SPK_F3 || \
            g_int8_kernel_container[kid].ktype == CONV_2SPK_FN || g_int8_kernel_container[kid].ktype == CONV_2SPK_FS ) {

	    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_int8_kernel_container[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;
    
        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                num_chl_per_grp_pad, conv_param.num_grp, g_int8_kernel_container[kid].tile_k_per_cta, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                g_int8_kernel_container[kid].tile_k_per_cta, pad_size);

        if(splitk == 1) {
            (g_int8_kernel_container[kid].int8_lut_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_LUT_KPARAM_LIST);
        }
	    else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;
            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_int8_kernel_container[kid].tile_k_per_cta, splitk);

            (g_int8_kernel_container[kid].int8_spk_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_SPK_KPARAM_LIST);
        }
    }
    
    if(splitk > 1 || splitf > 1) {
        int spk_width_v4   = num_flt_per_grp_pad * conv_param.num_grp / __INT4__;
        int spk_height_v1  = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x  = spk_height_v1;
        merge_grid_size.y  = DivUp(spk_width_v4, merge_block_size.x);
        merge_grid_size.z  = 1;

        MergeConvSplitResultsFp32<<<merge_grid_size, merge_block_size, 0, stream>>>(INT8_MERGE_KPARAM_LIST);
    }

    if(is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
    
}

/* -----------------  JIT INT8 KERNEL ------------------ */

#define MAX_KERNEL_SIZE (1 + 12 + 30)

__inline__ std::string ToString(int v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

ppl::common::RetCode PPLCUDAConvolutionPredictKernelInt8(
    ppl::common::datatype_t type,
    algo_param_t &algo_param,
    conv_param_t &conv_param)
{
    int out_hw      = conv_param.in_num * conv_param.out_height * conv_param.out_width;
    int flt_hw      = conv_param.flt_height * conv_param.flt_width;
    int chl_per_grp = conv_param.num_chl / conv_param.num_grp;

    if (out_hw < 32) {
        algo_param.tiles.m_cta = 16;
    } else if (out_hw < 512) {
        algo_param.tiles.m_cta = 32;
    } else {
        algo_param.tiles.m_cta = 64;
    }

    if (conv_param.num_flt < 16) {
        algo_param.tiles.n_cta = 8;
    } else if (conv_param.num_flt < 64) {
        algo_param.tiles.n_cta = 16;
    } else if (conv_param.num_flt < 128) {
        algo_param.tiles.n_cta = 32;
    } else {
        algo_param.tiles.n_cta =64;
    }
        
    if (chl_per_grp < 32) {
        algo_param.tiles.k_cta = 16;
    } else if (chl_per_grp == 32) {
        algo_param.tiles.k_cta = 32;
    } else if (chl_per_grp < 200) {
        algo_param.tiles.k_cta = 64;
    } else if (chl_per_grp < 1024) {
        algo_param.tiles.k_cta = 128;
    } else {
        algo_param.tiles.k_cta = 256;
    }

    algo_param.tiles.m_warp = algo_param.tiles.m_cta;
    algo_param.tiles.n_warp = algo_param.tiles.n_cta;
    algo_param.tiles.k_per_set = algo_param.tiles.k_cta;

    if (algo_param.tiles.k_cta > 128) {
        algo_param.tiles.k_per_set /= 4;
    } else if (algo_param.tiles.k_cta > 64) {
        algo_param.tiles.k_per_set /= 2;
        algo_param.tiles.m_warp /= 2;
        algo_param.tiles.n_warp /= 2;
    } else {
        algo_param.tiles.m_warp /= 2;
        algo_param.tiles.n_warp /= 2;
    }

    if (algo_param.tiles.m_warp < 8)  algo_param.tiles.m_warp = 8;
    if (algo_param.tiles.n_warp < 8)  algo_param.tiles.n_warp = 8;

    int cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) * (algo_param.tiles.n_cta / algo_param.tiles.n_warp) * WARP_SIZE;
    int chl_per_grp_pad = Align(chl_per_grp, 4);
    int kloop_num  = DivUp(flt_hw * chl_per_grp_pad, algo_param.tiles.k_cta);
    algo_param.tiles.k_per_step = (kloop_num * algo_param.tiles.k_cta * 4) / cta_size_in_thd;
    for (int32_t i = 32; i <= 128; i *= 2) {
        if (i == 128 || i > algo_param.tiles.k_per_step) {
            algo_param.tiles.k_per_step = i >> 1;
            break;
        }
    }
    return ppl::common::RC_SUCCESS;
}

float AlgoForwardTimeInt8(
    cudaStream_t &stream,
    std::vector<string> name,
    string code,
    int &idx,
    std::vector<const char *> compile_params,
    int device,
    bool include,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    std::vector<algo_param_t> &algo_param,
    conv_param_t &conv_param,
    quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    uint64_t workspace)
{
    float elapsed = 0;

#ifdef PPLNN_ENABLE_CUDA_JIT
    std::string src_name                   = name[0];
    string ptx                             = ppl::nn::cuda::CUDANVRTCCompile(pair<string, string>(src_name, code), compile_params, device, include);
    ppl::nn::cuda::CUDAModule *cuda_module = new ppl::nn::cuda::CUDAModule();
    cuda_module->SetSourceCode(src_name, ptx);
    float min_time = FLT_MAX;
    int times      = 1;

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    for (size_t n = 0; n < name.size(); n++) {
        CUfunction function = cuda_module->GetKernelFunc(name[n]);
        cudaEventRecord(begin, stream);
        for (int i = 0; i < times; i++) {
            PPLCUDAConvolutionForwardJitImpInt8(
                stream, function, type, d_input, d_flt, d_output, bias, d_temp_buf, algo_param[n], conv_param, quant_param, fuse_param);
        }
        cudaEventRecord(end, stream);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, begin, end);
        if (elapsed < min_time) {
            min_time = elapsed;
            idx      = n;
        }
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    delete cuda_module;
#endif
    return elapsed;
}

double PPLCUDAConvolutionJitSelectKernelInt8(
    int device_id,
    cudaStream_t &stream,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    uint64_t workspace)
{
    auto pre_algo_param     = algo_param;
    int pad_size            = GetPadSize(type);
    int num_chl_per_grp     = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp     = conv_param.num_flt / conv_param.num_grp;
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);
    int flt_hw              = conv_param.flt_height * conv_param.flt_width;

    std::vector<std::string> knames;
    std::vector<algo_param_t> params;
    std::string total_source = "";
    int declare_times        = 0;
    float elapsed;

    const int SPLITK_OPTIONS[] = {1, 2, 4, 8};
    for (unsigned int spf = 0; spf < 2; spf++) {
        for (unsigned int spk = 0; spk < 4; spk++) {
            unsigned int splitk = SPLITK_OPTIONS[spk];
            unsigned int splitf = spf ? flt_hw : 1;

            if (spf == 1 && flt_hw == 1)
                continue;
            if (spf >= 1 && spk >= 1)
                continue;

            for (unsigned int index = 0; index < MAX_KERNEL_SIZE * 2; index++) {
                conv_ktype_t ktype;
                algo_param = pre_algo_param;
                PPLCUDAConvolutionModifyAlgoParam(algo_param, index % MAX_KERNEL_SIZE); // change algo_param
                algo_param.splitk = splitk;
                algo_param.splitf = splitf;

                if ((spf >= 1 || spk >= 1) && index < MAX_KERNEL_SIZE) // jump idnx kernel when use splitk and splitf
                    continue;

                int size_x    = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, algo_param.tiles.m_cta);
                int size_y    = DivUp(num_flt_per_grp_pad, algo_param.tiles.n_cta);
                int grid_size = size_x * size_y * conv_param.num_grp;

                if (index < MAX_KERNEL_SIZE) { // Use non-shared memory algo for small channel
                    algo_param.tiles.flt_pad_size = algo_param.tiles.k_per_step / 4;
                    if (algo_param.tiles.k_per_step <= 16) {
                        ktype = CONV_IDXN_C2;
                    } else if (algo_param.tiles.k_per_step <= 32) {
                        ktype = CONV_IDXN_C4;
                    } else {
                        ktype = CONV_IDXN_C32;
                    }
                    algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                                    (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                                    WARP_SIZE;
                    algo_param.algo_name = "nvIdxnConv_imma8816_nhwc_b" + ToString(algo_param.tiles.m_cta) + "x" + ToString(algo_param.tiles.n_cta) +
                                        "_w" + ToString(algo_param.tiles.m_warp) + "x" + ToString(algo_param.tiles.n_warp) +
                                        "_k" + ToString(algo_param.tiles.k_cta) + "_s" + ToString(algo_param.tiles.k_per_step) + "_nosmem";
                } else { // Use 2spk algo for large channel
                    algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                                    (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                                    (algo_param.tiles.k_cta / algo_param.tiles.k_per_set) *
                                                    WARP_SIZE;
                    ktype                     = CONV_2SPK_FN;
                    std::string f_size        = "fn";
                    algo_param.tiles.flt_size = 0;
                    if (spf) {
                        ktype                     = CONV_2SPK_FS;
                        f_size                    = "fs";
                        algo_param.tiles.flt_size = 1;
                    } else if (conv_param.flt_height == 1 && conv_param.flt_width == 1) {
                        ktype                     = CONV_2SPK_F1;
                        f_size                    = "f1";
                        algo_param.tiles.flt_size = 1;
                    } else if (conv_param.flt_height == 3 && conv_param.flt_width == 3) {
                        ktype                     = CONV_2SPK_F3;
                        f_size                    = "f3";
                        algo_param.tiles.flt_size = 3;
                    }
                    algo_param.algo_name = "nv2spkConv_imma8816_nhwc_" + f_size + "_b" + ToString(algo_param.tiles.m_cta) + "x" + ToString(algo_param.tiles.n_cta) +
                                        "_w" + ToString(algo_param.tiles.m_warp) + "x" + ToString(algo_param.tiles.n_warp) +
                                        "_k" + ToString(algo_param.tiles.k_cta) + "_s" + ToString(algo_param.tiles.k_per_set) + "_buf1";
                    if (splitk > 1) {
                        algo_param.algo_name = algo_param.algo_name + "_splitk";
                    }
                }
                kernel_info_t temp_kernel(-1, ktype, algo_param.algo_name.c_str());
                if (!temp_kernel.CheckKernelTilesFeasible(type, device_id))
                    continue;
                if (!temp_kernel.CheckKernelTypeFeasibleInt8(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk))
                    continue;
                if (!temp_kernel.CheckSplitkFeasible(num_chl_per_grp, splitk))
                    continue;
                if (!temp_kernel.CheckSplitfFeasible(splitf, splitk))
                    continue;
                if (!temp_kernel.CheckQuickSelectFeasible(algo_param, num_chl_per_grp, grid_size, flt_hw, splitk, splitf, device_id))
                    continue;

                auto mgr = CodeGeneFactorManager::Instance();
                auto gene_factor = mgr->FindKernel(type);
                std::string source = "";
                fuse_info_t empty_fuse_info;
                if (algo_param.algo_name.find("Idxn") != std::string::npos) {
                    gene_factor->GeneIdxnKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta, algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta, algo_param.tiles.k_per_step, declare_times);
                    gene_factor->ReplaceFusionForIdxn(source, empty_fuse_info);
                    declare_times++;
                } else if (algo_param.algo_name.find("2spk") != std::string::npos) {
                    gene_factor->Gene2spkKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta, algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta, algo_param.tiles.k_per_set, algo_param.splitk, algo_param.splitf, algo_param.tiles.buf, declare_times);
                    gene_factor->ReplaceFusionFor2spk(source, empty_fuse_info);
                    declare_times++;
                }

                if (std::find(knames.begin(), knames.end(), algo_param.algo_name) == knames.end()) {
                    total_source = total_source + source;
                }
                knames.push_back(algo_param.algo_name);
                params.push_back(algo_param);
            }
        }
    }
    int index = 0;
    std::vector<const char *> compile_params;
    elapsed = AlgoForwardTimeInt8(stream, knames, total_source, index, compile_params, device_id, true, type, d_input, d_flt, d_output, bias, d_temp_buf, params, conv_param, quant_param, fuse_param, workspace);

    algo_param                         = params[index];
    return elapsed;
}

void PPLCUDAConvolutionForwardJitImpInt8(
    cudaStream_t &stream,
    CUfunction function,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    quant_param_t &quant_param,
    fuse_param_t &fuse_param)
{
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input  = d_input;
    int4 *pad_output = d_output;

    if (is_in_grp_pad) {
        pad_input = d_temp_buf;
        buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if (is_out_grp_pad) {
        pad_output = d_temp_buf + buf_off_v4;
        buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    }

    int4 *final_out  = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    float clip_min     = fuse_param.clip_min;
    float clip_max     = fuse_param.clip_max;
    float elt_clip_min = fuse_param.elt_clip_min;
    float elt_clip_max = fuse_param.elt_clip_max;
    float leaky        = fuse_param.leaky;
    float elt_leaky    = fuse_param.elt_leaky;
    
    const int4 *pre_data  = (const int4 *)fuse_param.pre_data;
    const void *prelu     = (const void *)fuse_param.prelu;
    const void *elt_prelu = (const void *)fuse_param.elt_prelu;

    int tile_n = algo_param.tiles.n_cta;
    int tile_m = algo_param.tiles.m_cta;
    int cta_k  = algo_param.tiles.k_cta;

    dim3 block_size, grid_size;
    block_size.x = algo_param.tiles.cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_m);
    grid_size.y = DivUp(num_flt_per_grp_pad, tile_n);
    grid_size.z = conv_param.num_grp * splitk * splitf;
    if (algo_param.algo_name.find("Idxn") != std::string::npos) {
        int img_pad_size = pad_size;
        int flt_pad_size = algo_param.tiles.flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

        int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, cta_k);
        int koff_num_pad = Align(kloop_num * (cta_k / flt_pad_size), WARP_SIZE);

        void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &koff_num_pad, &in_hw, &out_hw, &flt_hw, &out_nhw, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &conv_param.num_chl, &num_chl_per_grp, &in_chl_per_grp_pad, &flt_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &quant_param.in_scale, &quant_param.d_flt_scale, &quant_param.out_scale, &quant_param.pre_scale, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};

        CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
    } else if (algo_param.algo_name.find("2spk") != std::string::npos) {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);
        if (splitk == 1) {
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &quant_param.in_scale, &quant_param.d_flt_scale, &quant_param.out_scale, &quant_param.pre_scale, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;
            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, cta_k, splitk);
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &num_chl_per_spk_head, &num_chl_per_spk_tail, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &quant_param.in_scale, &quant_param.d_flt_scale};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
        }
    }

    if(splitk > 1 || splitf > 1) {
        int spk_width_v4   = num_flt_per_grp_pad * conv_param.num_grp / __INT4__;
        int spk_height_v1  = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x  = spk_height_v1;
        merge_grid_size.y  = DivUp(spk_width_v4, merge_block_size.x);
        merge_grid_size.z  = 1;

        MergeConvSplitResultsFp32<<<merge_grid_size, merge_block_size, 0, stream>>>(INT8_MERGE_KPARAM_LIST);
    }

    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
}
