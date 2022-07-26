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
#include "conv_jit.h"
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

#define INT8_SWZL_SPK_KPARAM_LIST                          \
        d_flt,                                             \
        pad_input,                                         \
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


#define INT8_SWZL_LUT_KPARAM_LIST \
            d_flt,                                                                      \
            pad_input,                                                                  \
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

static std::vector<kernel_info_t> g_int8_kvec;
static bool is_g_int8_kvec_initialized = false;

static std::unordered_map<size_t, algo_param_t> g_conv_shape_hash;

__inline__ void InitializeInt8ConvKernelContainer(std::vector<kernel_info_t> &g_int8_kvec, int device_id, ppl::common::datatype_t type)
{
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    if(type == ppl::common::DATATYPE_INT8) {
#ifndef PPLNN_ENABLE_CUDA_JIT
        if (device_prop.major == 7 && device_prop.minor == 5) {
            // sm75 kernels
            Initialize2spkSM75Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM75Int8Imma8816ConvF3KernelContainer(g_int8_kvec);
            Initialize2spkSM75Int8Imma8816ConvFNKernelContainer(g_int8_kvec);
            Initialize2spkSM75Int8Imma8816ConvFSKernelContainer(g_int8_kvec);

            InitializeIdxnSM75Int8Imma8816ConvKernelContainer(g_int8_kvec);

            InitializeSwzlSM75Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            InitializeSwzlSM75Int8Imma8816ConvF3KernelContainer(g_int8_kvec);
            InitializeSwzlSM75Int8Imma8816ConvFNKernelContainer(g_int8_kvec);
        } else if (device_prop.major > 8 || (device_prop.major == 8 && device_prop.minor >= 0)) {
            // sm75 kernels
            Initialize2spkSM75Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM75Int8Imma8816ConvF3KernelContainer(g_int8_kvec);
            Initialize2spkSM75Int8Imma8816ConvFNKernelContainer(g_int8_kvec);
            Initialize2spkSM75Int8Imma8816ConvFSKernelContainer(g_int8_kvec);

            InitializeIdxnSM75Int8Imma8816ConvKernelContainer(g_int8_kvec);

            InitializeSwzlSM75Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            InitializeSwzlSM75Int8Imma8816ConvF3KernelContainer(g_int8_kvec);
            InitializeSwzlSM75Int8Imma8816ConvFNKernelContainer(g_int8_kvec);

            // sm80 kernels
            Initialize2spkSM80Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma8816ConvF3KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma8816ConvFNKernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma8816ConvFSKernelContainer(g_int8_kvec);

            InitializeSwzlSM80Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            InitializeSwzlSM80Int8Imma8816ConvF3KernelContainer(g_int8_kvec);
            InitializeSwzlSM80Int8Imma8816ConvFNKernelContainer(g_int8_kvec);

            Initialize2spkSM80Int8Imma16816ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16816ConvF3KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16816ConvFNKernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16816ConvFSKernelContainer(g_int8_kvec);

            Initialize2spkSM80Int8Imma16832ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16832ConvF3KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16832ConvFNKernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16832ConvFSKernelContainer(g_int8_kvec);

            InitializeIdxnSM80Int8Imma16816ConvKernelContainer(g_int8_kvec);

            InitializeSwzlSM80Int8Imma16816ConvF1KernelContainer(g_int8_kvec);
            InitializeSwzlSM80Int8Imma16816ConvF3KernelContainer(g_int8_kvec);
            InitializeSwzlSM80Int8Imma16816ConvFNKernelContainer(g_int8_kvec);

            InitializeSwzlSM80Int8Imma16832ConvF1KernelContainer(g_int8_kvec);
            InitializeSwzlSM80Int8Imma16832ConvF3KernelContainer(g_int8_kvec);
            InitializeSwzlSM80Int8Imma16832ConvFNKernelContainer(g_int8_kvec);
        }
#endif
    }

    is_g_int8_kvec_initialized = true;
}

__inline__ size_t GetConvShapeHashKey(conv_param_t &conv_param)
{
    return std::hash<std::string>{}(GetConvShapeString(conv_param));
}

/* -----------------  INT8 KERNEL ------------------ */

double PPLCUDAConvolutionSelectKernelInt8(
        int device_id,
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
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    if(!is_g_int8_kvec_initialized)
        InitializeInt8ConvKernelContainer(g_int8_kvec, device_id, type);

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

        for(unsigned int kid = 0; kid < g_int8_kvec.size(); kid++) {
            unsigned int splitf = (g_int8_kvec[kid].ktype == CONV_2SPK_FS) ? flt_hw : 1;

            if(!g_int8_kvec[kid].CheckKernelTypeFeasibleInt8(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk)) continue;

            if (!g_int8_kvec[kid].CheckSMemSizeFeasible(device_prop))
                continue;

            if (!g_int8_kvec[kid].CheckGpuArchFeasible(device_prop))
                continue;

            if(!g_int8_kvec[kid].CheckSplitkFeasible(num_chl_per_grp, splitk))
                continue;

            if(!g_int8_kvec[kid].CheckSplitfFeasible(splitf, splitk))
                continue;

            int4 *conv_out = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

            dim3 block_size, grid_size;

            block_size.x = g_int8_kvec[kid].cta_size_in_thd;
            block_size.y = 1;
            block_size.z = 1;

            int smem_size = g_int8_kvec[kid].smem_size;

            if(g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || \
                    g_int8_kvec[kid].ktype == CONV_SWZL_FN) {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_int8_kvec[kid].tile_n_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_int8_kvec[kid].tile_m_per_cta);
            } else {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_int8_kvec[kid].tile_m_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_int8_kvec[kid].tile_n_per_cta);
            }

            grid_size.z = conv_param.num_grp * splitk * splitf;

	        cudaEventRecord(begin, stream);

	        for(int i = 0; i < TIMES; i++) {
                if(g_int8_kvec[kid].ktype == CONV_IDXN_C4 || g_int8_kvec[kid].ktype == CONV_IDXN_C8 || \
                        g_int8_kvec[kid].ktype == CONV_IDXN_C64) {
                    int tile_k_per_step = g_int8_kvec[kid].tile_k_per_step;

                    int img_pad_size    = pad_size;
                    int flt_pad_size    = g_int8_kvec[kid].flt_pad_size;
                    int out_nhw         = out_hw * conv_param.in_num;

                    int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
                    int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
                    int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	                int kloop_num        = DivUp(flt_hw * flt_chl_per_grp_pad, g_int8_kvec[kid].tile_k_per_cta);
                    int koff_num_pad      = Align(kloop_num * (g_int8_kvec[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

                    (g_int8_kvec[kid].int8_idx_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_IDX_KPARAM_LIST);
                }
                else if(g_int8_kvec[kid].ktype == CONV_2SPK_F1 || g_int8_kvec[kid].ktype == CONV_2SPK_F3 || \
                        g_int8_kvec[kid].ktype == CONV_2SPK_FN || g_int8_kvec[kid].ktype == CONV_2SPK_FS || \
                        g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || \
                        g_int8_kvec[kid].ktype == CONV_SWZL_FN) {

	                int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_int8_kvec[kid].tile_k_per_cta);

                    lut_t in_lut, flt_lut;
                    int in_lut_size, flt_lut_size;
                
                    InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                            conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                            num_chl_per_grp_pad, conv_param.num_grp, g_int8_kvec[kid].tile_k_per_cta, pad_size);

                    InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                            g_int8_kvec[kid].tile_k_per_cta, pad_size);


                    if(splitk == 1) {
                        g_int8_kvec[kid].AdaptInt8LutKernelSMemSize();

                        if(g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || g_int8_kvec[kid].ktype == CONV_SWZL_FN)
                            (g_int8_kvec[kid].int8_lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_SWZL_LUT_KPARAM_LIST);
                        else {
                            (g_int8_kvec[kid].int8_lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_LUT_KPARAM_LIST);
                        }
                    } else {
                        int num_chl_per_spk_head, num_chl_per_spk_tail;
                        InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_int8_kvec[kid].tile_k_per_cta, splitk);

                        g_int8_kvec[kid].AdaptInt8SpkKernelSMemSize();

                        if(g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || g_int8_kvec[kid].ktype == CONV_SWZL_FN)
                            (g_int8_kvec[kid].int8_spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_SWZL_SPK_KPARAM_LIST);
                        else {
                            (g_int8_kvec[kid].int8_spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_SPK_KPARAM_LIST);
                        }
    
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
        int device_id,
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
    if(!is_g_int8_kvec_initialized)
        InitializeInt8ConvKernelContainer(g_int8_kvec, device_id, type);

    unsigned int kid = algo_param.kid;
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

    block_size.x = g_int8_kvec[kid].cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    int smem_size = g_int8_kvec[kid].smem_size;

    if(g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || \
            g_int8_kvec[kid].ktype == CONV_SWZL_FN) {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_int8_kvec[kid].tile_n_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_int8_kvec[kid].tile_m_per_cta);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_int8_kvec[kid].tile_m_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_int8_kvec[kid].tile_n_per_cta);
    }

    grid_size.z  = conv_param.num_grp * splitk * splitf;

    if(g_int8_kvec[kid].ktype == CONV_IDXN_C4 || g_int8_kvec[kid].ktype == CONV_IDXN_C8 || \
            g_int8_kvec[kid].ktype == CONV_IDXN_C64) {
        int img_pad_size = pad_size;
        int flt_pad_size = g_int8_kvec[kid].flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	    int kloop_num = DivUp(flt_hw * flt_chl_per_grp_pad, g_int8_kvec[kid].tile_k_per_cta);
        int koff_num_pad = Align(kloop_num * (g_int8_kvec[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

        (g_int8_kvec[kid].int8_idx_kptr)<<<grid_size, block_size, 0, stream>>>(INT8_IDX_KPARAM_LIST);

    } else if(g_int8_kvec[kid].ktype == CONV_2SPK_F1 || g_int8_kvec[kid].ktype == CONV_2SPK_F3 || \
              g_int8_kvec[kid].ktype == CONV_2SPK_FN || g_int8_kvec[kid].ktype == CONV_2SPK_FS || \
              g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || \
              g_int8_kvec[kid].ktype == CONV_SWZL_FN) {

	    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_int8_kvec[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;
    
        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                num_chl_per_grp_pad, conv_param.num_grp, g_int8_kvec[kid].tile_k_per_cta, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                g_int8_kvec[kid].tile_k_per_cta, pad_size);

        if(splitk == 1) {
            g_int8_kvec[kid].AdaptInt8LutKernelSMemSize();

            if(g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || g_int8_kvec[kid].ktype == CONV_SWZL_FN)
                (g_int8_kvec[kid].int8_lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_SWZL_LUT_KPARAM_LIST);
            else {
                (g_int8_kvec[kid].int8_lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_LUT_KPARAM_LIST);
            }
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;
            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_int8_kvec[kid].tile_k_per_cta, splitk);

            g_int8_kvec[kid].AdaptInt8SpkKernelSMemSize();

            if(g_int8_kvec[kid].ktype == CONV_SWZL_F1 || g_int8_kvec[kid].ktype == CONV_SWZL_F3 || g_int8_kvec[kid].ktype == CONV_SWZL_FN)
                (g_int8_kvec[kid].int8_spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_SWZL_SPK_KPARAM_LIST);
            else
                (g_int8_kvec[kid].int8_spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_SPK_KPARAM_LIST);
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

float AlgoForwardTimeInt8(
    int device_id,
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
                device_id, stream, function, type, d_input, d_flt, d_output, bias, d_temp_buf, algo_param[n], conv_param, quant_param, fuse_param);
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

ppl::common::RetCode GetInt8ConvKernelNominees(
    int device_id,
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    std::vector<std::string> & knames,
    std::vector<algo_param_t> & params,
    std::string & sources)
{
#ifdef PPLNN_ENABLE_CUDA_JIT
    int pad_size            = GetPadSize(type);
    int num_grp             = conv_param.num_grp;
    int num_chl_per_grp     = conv_param.num_chl / num_grp;
    int num_flt_per_grp     = conv_param.num_flt / num_grp;
    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    // int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int batch               = conv_param.in_num;
    int flt_h               = conv_param.flt_height;
    int flt_w               = conv_param.flt_width;
    int flt_hw              = flt_h * flt_w;
    // int in_hw               = conv_param.in_height  * conv_param.in_width;
    int out_w               = conv_param.out_width;
    int out_hw              = conv_param.out_height * conv_param.out_width;

    int type_size = ppl::common::GetSizeOfDataType(type);

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    int m_conv = Align(batch * out_hw,  pad_size);
    int n_conv = Align(num_flt_per_grp, pad_size);

    int sm_num = device_prop.multiProcessorCount;
    int device_arch      = device_prop.major * 10 + device_prop.minor;
    int max_regs_per_thd = 255;
    int max_regs_per_sm  = device_prop.regsPerMultiprocessor;
    int max_ctas_per_sm  = device_prop.maxBlocksPerMultiProcessor;
    int max_thds_per_sm  = device_prop.maxThreadsPerMultiProcessor;
    int max_smem_per_sm  = device_prop.sharedMemPerMultiprocessor;
    int max_regs_per_cta = device_prop.regsPerBlock;
    int max_smem_per_cta = device_prop.sharedMemPerBlock;
    int max_dyn_smem_per_cta = 0;

    std::string mma_shape = "";
    int splitk = 1;
    int splitf = 1;
    int m_mma = 0;
    int n_mma = 0;
    int k_mma = 0;
    int m_mma_max = 0;
    int n_mma_max = 0;
    int k_mma_max = 0;
    int k_blk_mma = 0;
    int buf_num_max = 0;

    int cpi_mma = 0;
    int cpi_ldg32_l1d = 0;
    int cpi_ldg64_l1d = 0;
    int cpi_ldg128_l1d = 0;
    int cpi_ldg32_l2 = 0;
    int cpi_ldg64_l2 = 0;
    int cpi_ldg128_l2 = 0;
    int cpi_lds32 = 0;
    int cpi_lds64 = 0;
    int cpi_lds128 = 0;
    int cpi_sts32 = 0;
    int cpi_sts64 = 0;
    int cpi_sts128 = 0;
    int latency_mma = 0;
    int latency_l2_cache = 0;
    int latency_dram = 0;

    std::vector<std::pair<algo_param_t, float>> nominees;
    algo_param_t nominee;
    nominee.algo_type = "TuringIMMAImpgemm";

    GetHardwareInfo(device_arch, type, num_chl_per_grp, cpi_mma, latency_mma, cpi_ldg32_l1d, cpi_ldg64_l1d, \
            cpi_ldg128_l1d, cpi_ldg32_l2, cpi_ldg64_l2, cpi_ldg128_l2, cpi_lds32, cpi_lds64, cpi_lds128, \
            cpi_sts32, cpi_sts64, cpi_sts128, latency_l2_cache, latency_dram, max_dyn_smem_per_cta);

    if (num_chl_per_grp <= 64) {
        int k_per_step = 0;
        if (num_chl_per_grp > 0 && num_chl_per_grp <= 4) {
            k_per_step = 16;
        } else if (num_chl_per_grp > 4 && num_chl_per_grp <= 8) {
            k_per_step = 32;
        } else if (num_chl_per_grp > 8 && num_chl_per_grp <= 64) {
            k_per_step = 64;
        }
        
        GetIdxnMmaInfo(device_arch, type, num_chl_per_grp, mma_shape, m_mma, n_mma, k_mma, m_mma_max, n_mma_max, k_mma_max);

        int flt_pad_size = k_per_step >> 2;
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int k_conv = flt_hw * flt_chl_per_grp_pad;

        // loop over idxn kernel space
        for(int k_num = 1; k_num <= 2; k_num *= 2)
            for(int m_warp = m_mma; m_warp <= m_mma_max; m_warp *= 2)
                for(int n_warp = n_mma; n_warp <= n_mma_max; n_warp *= 2)
                    for(int m_warp_num = 1; m_warp_num <= 4; m_warp_num *= 2)
                        for(int n_warp_num = 1; n_warp_num <= 4; n_warp_num *= 2) {

                            int m_cta = m_warp * m_warp_num;
                            int n_cta = n_warp * n_warp_num;
                            int k_cta = k_per_step * k_num;
                            int cta_size_in_warp = m_warp_num * n_warp_num;
                            int cta_size_in_thd  = cta_size_in_warp * WARP_SIZE;

                            int kloop_total  = DivUp(flt_hw * flt_chl_per_grp_pad, k_cta);
                            int kloop_num = kloop_total;

                            int m_cta_num = DivUp(m_conv, m_cta);
                            int n_cta_num = DivUp(n_conv, n_cta);
                            int cta_num = m_cta_num * n_cta_num;

                            // filter out cases with too large tiles
                            if( m_warp_num == 4 && n_warp_num == 4 ) continue;
                            if(m_warp == m_mma_max && n_warp == n_mma_max) continue;

                            // filter out cases that kloop_time > 1
                            int kloop_time = DivUp(kloop_num * (k_cta / flt_pad_size), cta_size_in_thd);
                            if(kloop_time != 1) continue;

                            // filter out cases with too much register usage
                            int regs_per_thd = GetIdxnRegsPerThread(type, m_cta, n_cta, m_warp, n_warp, k_per_step, m_mma, n_mma, k_mma, cta_size_in_thd);
                            int regs_per_cta = regs_per_thd * cta_size_in_thd;
                            if (regs_per_thd > max_regs_per_thd) continue;
                            if (regs_per_cta > max_regs_per_cta) continue; 

                            // filter out cases with too much smem usage
                            int smem_per_cta = GetIdxnSmemUsage(m_cta, cta_size_in_thd);
                            if (smem_per_cta > max_smem_per_cta) continue;

                            // filter out cases with too much padding
                            float eff_score = GetEfficiencyScore(m_cta, n_cta, k_cta, kloop_total, m_conv, n_conv, k_conv);
                            if(eff_score < 0.5) continue;

                            // filter out cases with too low occupancy
                            float cta_launch_times = 0.f;
                            float occ_score = GetOccupancyScore(cta_size_in_thd, cta_size_in_warp, sm_num, cta_num, regs_per_cta, smem_per_cta, \
                                    max_ctas_per_sm, max_thds_per_sm, max_regs_per_sm, max_smem_per_sm, cta_launch_times);
                            if(occ_score < 0.5) continue;

                            // get kernel pipeline score
                            float pip_score = GetIdxnPipelineScore(type_size, cta_launch_times, out_w, cta_size_in_thd, cta_size_in_warp, m_cta, n_cta, k_cta, m_warp, n_warp, \
                                    k_per_step, m_mma, n_mma, k_mma, cpi_mma, cpi_ldg32_l1d, cpi_ldg64_l1d, cpi_ldg128_l1d, cpi_ldg32_l2, \
                                    cpi_ldg64_l2, cpi_ldg128_l2, latency_mma, latency_l2_cache, latency_dram);

                            // insert one nominee
                            float score = eff_score + occ_score + pip_score;
                            nominee.SetIdxnKernelParam(m_cta, n_cta, k_cta, m_warp, n_warp, k_per_step, flt_pad_size, cta_size_in_thd, smem_per_cta, splitk, splitf, mma_shape);

                            nominees.push_back(std::make_pair(nominee, score));
                            // printf("insert nominee %s : eff %.2f occ %.2f pip %.2f launch %.2f\n", nominee.algo_name.c_str(), eff_score, occ_score, pip_score, cta_launch_times);
                        }

        if(nominees.size() == 0) { // insert default kernel
            // nvIdxnConv_b128x128_w64x64
            nominee.SetIdxnKernelParam(128, 128, k_per_step, 64, 64, k_per_step, flt_pad_size, 128, 4096, 1, 1, mma_shape);
            nominees.push_back(std::make_pair(nominee, 0.f));
        }

    } else {
        int flt_size = 0;
        int k_conv = flt_hw * num_chl_per_grp_pad;
        int estimate_cta_num = GetEstimateCtaNumber(m_conv, n_conv, num_grp);

        if(conv_param.flt_height == 1 && conv_param.flt_width == 1)
            flt_size = 1;
        else if(conv_param.flt_height == 3 && conv_param.flt_width == 3)
            flt_size = 3;
        else
            flt_size = 0;

        if(estimate_cta_num <= sm_num) { // choose 2spk kernel

            Get2spkMmaInfo(device_arch, type, mma_shape, m_mma, n_mma, k_mma, m_mma_max, n_mma_max, k_mma_max, k_blk_mma, buf_num_max);
            
            const int SPLITK_OPTIONS[] = {1, 2, 4, 8};

            // loop over 2spk kernel space
            for(int spf = 0; spf < 2; spf++)
                for(int spk = 0; spk < 4; spk++)
                    for(int buf_num = 1; buf_num <= buf_num_max; buf_num++)
                        for(int k_per_set = k_mma; k_per_set <= k_mma_max; k_per_set *= 2)
                            for(int set_num = 1; set_num <= 4; set_num *= 2)
                                for(int m_warp = m_mma; m_warp <= m_mma_max; m_warp *= 2)
                                    for(int n_warp = n_mma; n_warp <= n_mma_max; n_warp *= 2)
                                        for(int m_warp_num = 1; m_warp_num <= 4; m_warp_num *= 2)
                                            for(int n_warp_num = 1; n_warp_num <= 4; n_warp_num *= 2) {

                                                int m_cta = m_warp * m_warp_num;
                                                int n_cta = n_warp * n_warp_num;
                                                int k_cta = k_per_set * set_num;
                                                int set_size_in_warp = m_warp_num * n_warp_num;
                                                int cta_size_in_warp = m_warp_num * n_warp_num * set_num;
                                                int set_size_in_thd  = set_size_in_warp * WARP_SIZE;
                                                int cta_size_in_thd  = cta_size_in_warp * WARP_SIZE;

                                                // filter out kernels that is not aligned
                                                if( n_conv >= 128 && n_cta < 128 ) continue;

                                                // filter out splitk
                                                splitk = SPLITK_OPTIONS[spk];
                                                if( splitk > 1 && splitk * k_cta >= Align(num_chl_per_grp, k_cta) ) continue;

                                                // filter out splitf
                                                if(spf == 1) { splitf = flt_hw; flt_size = 11; }
                                                if(spf == 1 && splitf == 1) continue;
                                                if(splitk * splitf >= MAX_SPLIT_SIZE) continue;

                                                int m_cta_num = DivUp(m_conv, m_cta);
                                                int n_cta_num = DivUp(n_conv, n_cta);
                                                int cta_num = m_cta_num * n_cta_num * num_grp * splitk * splitf;

                                                int split = splitk * splitf;
                                                int kloop_total = flt_hw * DivUp(num_chl_per_grp_pad, k_cta);
                                                int kloop_num = kloop_total / split;

                                                // filter out too large and too small k_cta
                                                if( k_cta != GetTileKSize(num_chl_per_grp_pad, kloop_num) ) continue;

                                                // filter out cases with too large tiles
                                                if( m_warp_num == 4 && n_warp_num == 4 ) continue;
                                                if(m_warp == m_mma_max && n_warp == n_mma_max) continue;
                                                if(cta_size_in_thd == 32 && k_cta == 128) continue;
                                                if(cta_size_in_thd <= 64 && k_cta == 256) continue;
                                                if(cta_size_in_thd <= 128 && k_cta == 512) continue;
                                                if(buf_num > kloop_num) continue;

                                                // filter out cases with too much register usage
                                                int regs_per_thd = Get2spkRegsPerThread(type, type_size, m_cta, n_cta, k_cta, m_warp, n_warp, k_per_set, \
                                                        m_mma, n_mma, k_mma, k_blk_mma, buf_num, cta_size_in_thd, set_size_in_thd);
                                                int regs_per_cta = regs_per_thd * cta_size_in_thd;
                                                if (regs_per_thd > max_regs_per_thd) continue;
                                                if (regs_per_cta > max_regs_per_cta) continue;

                                                // filter out cases with too much smem usage
                                                int smem_per_cta = Get2spkSmemUsage(type, type_size, m_cta, n_cta, k_cta, set_num, buf_num);
                                                if (smem_per_cta > max_dyn_smem_per_cta) continue;

                                                // filter out cases with too much padding
                                                float eff_score = GetEfficiencyScore(m_cta, n_cta, k_cta, kloop_total, m_conv, n_conv, k_conv);
                                                if(eff_score < 0.5) continue;

                                                // filter out cases with too low occupancy
                                                float cta_launch_times = 0.f;
                                                float occ_score = GetOccupancyScore(cta_size_in_thd, cta_size_in_warp, \
                                                        sm_num, cta_num, regs_per_cta, smem_per_cta,  max_ctas_per_sm, \
                                                        max_thds_per_sm, max_regs_per_sm, max_smem_per_sm, cta_launch_times);
                                                if(occ_score < 0.5) continue;

                                                // filter out too much split and too small splits
                                                if( cta_launch_times > 1 ) continue;

                                                // get kernel pipeline score
                                                float pip_score = Get2spkPipelineScore(type_size, cta_launch_times, m_conv, n_conv, k_conv, \
                                                        kloop_num, splitk, splitf, out_w, cta_size_in_thd, cta_size_in_warp, sm_num, m_cta, \
                                                        n_cta, k_cta, m_warp, n_warp, k_per_set, set_num, buf_num, m_mma, n_mma, k_mma, k_mma_max, \
                                                        cpi_mma, cpi_ldg128_l1d, cpi_ldg128_l2, cpi_lds128, cpi_sts32, latency_mma, \
                                                        latency_l2_cache, latency_dram);

                                                // insert one nominee
                                                float score = eff_score + occ_score + pip_score;
                                                // float score = pip_score;
                                                nominee.Set2spkKernelParam(m_cta, n_cta, k_cta, m_warp, n_warp, k_per_set, \
                                                        flt_size, buf_num, cta_size_in_thd, smem_per_cta, splitk, splitf, mma_shape);

                                                nominees.push_back(std::make_pair(nominee, score));
                                                // printf("insert 2spk nominee %s : eff %.2f occ %.2f pip %.2f launch %.2f cta_num %d warp_num %d\n",
                                                //         nominee.algo_name.c_str(), eff_score, occ_score, pip_score, cta_launch_times, cta_num, cta_size_in_warp);
                                            }

            if(nominees.size() == 0) { // insert default kernel
                if(conv_param.flt_height == 1 && conv_param.flt_width == 1)
                    flt_size = 1;
                else if(conv_param.flt_height == 3 && conv_param.flt_width == 3)
                    flt_size = 3;
                else
                    flt_size = 0;

                // nv2spkConv_b64x64_w32x32_k64_s64_buf1
                nominee.Set2spkKernelParam(64, 64, 64, 32, 32, 64, flt_size, 1, 128, 8192, 1, 1, mma_shape);
                nominees.push_back(std::make_pair(nominee, 0.f));
            }

        } else { // choose swzl kernels
            GetSwzlMmaInfo(device_arch, type, mma_shape, m_mma, n_mma, k_mma, m_mma_max, n_mma_max, k_mma_max, k_blk_mma, buf_num_max);

            // switch m_conv <=> n_conv
            int tmp = m_conv; m_conv = n_conv; n_conv = tmp;
            
            // loop over swzl kernel space
            for(int buf_num = 1; buf_num <= buf_num_max; buf_num++)
                for(int k_cta = k_mma; k_cta <= k_mma_max; k_cta *= 2)
                    for(int m_warp = m_mma; m_warp <= m_mma_max; m_warp *= 2)
                        for(int n_warp = n_mma; n_warp <= n_mma_max; n_warp *= 2)
                            for(int m_warp_num = 1; m_warp_num <= 4; m_warp_num *= 2)
                                for(int n_warp_num = 1; n_warp_num <= 4; n_warp_num *= 2) {

                                    int m_cta = m_warp * m_warp_num;
                                    int n_cta = n_warp * n_warp_num;
                                    int cta_size_in_warp = m_warp_num * n_warp_num;
                                    int cta_size_in_thd  = cta_size_in_warp * WARP_SIZE;

                                    // filter out kernels that is not aligned
                                    if( m_conv >= 64 && m_cta < 64 ) continue;

                                    int m_cta_num = DivUp(m_conv, m_cta);
                                    int n_cta_num = DivUp(n_conv, n_cta);
                                    int cta_num = m_cta_num * n_cta_num * num_grp;

                                    int kloop_total = flt_hw * DivUp(num_chl_per_grp_pad, k_cta);
                                    int kloop_num = kloop_total;

                                    // filter out too large and too small k_cta
                                    if( k_cta != GetTileKSize(num_chl_per_grp_pad, kloop_num) ) continue;

                                    // filter out cases with too large tiles
                                    if( m_warp == m_mma && n_warp == n_mma ) continue;
                                    if( m_warp == m_mma_max && n_warp == n_mma_max ) continue;
                                    if( m_warp_num == 4 && n_warp_num == 4 ) continue;
                                    if( m_warp_num == 1 && n_warp_num == 1 && k_cta == k_mma_max ) continue;
                                    if(buf_num > kloop_num) continue;

                                    // filter out cases with too much register usage
                                    int regs_per_thd = GetSwzlRegsPerThread(type, type_size, m_cta, n_cta, k_cta, m_warp, n_warp, \
                                            m_mma, n_mma, k_mma, k_blk_mma, buf_num, cta_size_in_thd);
                                    int regs_per_cta = regs_per_thd * cta_size_in_thd;
                                    if (regs_per_thd > max_regs_per_thd) continue;
                                    if (regs_per_cta > max_regs_per_cta) continue;

                                    // filter out cases with too much smem usage
                                    int smem_per_cta = GetSwzlSmemUsage(type, type_size, m_cta, n_cta, k_cta, m_warp, n_warp, \
                                            m_mma, n_mma, buf_num, cta_size_in_warp);
                                    if (smem_per_cta > max_dyn_smem_per_cta) continue;

                                    // filter out cases with too much padding
                                    float eff_score = GetEfficiencyScore(m_cta, n_cta, k_cta, kloop_total, m_conv, n_conv, k_conv);
                                    if(eff_score < 0.5) continue;

                                    // filter out cases with too low occupancy
                                    float cta_launch_times = 0.f;
                                    float occ_score = GetOccupancyScore(cta_size_in_thd, cta_size_in_warp, \
                                            sm_num, cta_num, regs_per_cta, smem_per_cta,  max_ctas_per_sm, \
                                            max_thds_per_sm, max_regs_per_sm, max_smem_per_sm, cta_launch_times);
                                    if(occ_score < 0.5) continue;

                                    // get kernel pipeline score
                                    float pip_score = GetSwzlPipelineScore(type_size, cta_launch_times, m_conv, n_conv, k_conv, \
                                            kloop_num, out_w, cta_size_in_thd, cta_size_in_warp, sm_num, m_cta, \
                                            n_cta, k_cta, m_warp, n_warp, buf_num, m_mma, n_mma, k_mma, k_mma_max, \
                                            cpi_mma, cpi_ldg128_l1d, cpi_ldg128_l2, cpi_lds128, cpi_sts32, latency_mma, \
                                            latency_l2_cache, latency_dram);

                                    // insert one nominee
                                    float score = eff_score + occ_score + pip_score;
                                    // float score = pip_score;
                                    nominee.SetSwzlKernelParam(m_cta, n_cta, k_cta, m_warp, n_warp, flt_size, \
                                            buf_num, cta_size_in_thd, smem_per_cta, splitk, splitf, mma_shape);

                                    nominees.push_back(std::make_pair(nominee, score));
                                    // printf("insert swzl nominee %s : eff %.2f occ %.2f pip %.2f launch %.2f cta_num %d warp_num %d\n",
                                    //         nominee.algo_name.c_str(), eff_score, occ_score, pip_score, cta_launch_times, cta_num, cta_size_in_warp);
                                }

            if(nominees.size() == 0) { // insert default kernel
                // nvswzlConv_b128x128_w64x64_k64_buf1
                nominee.SetSwzlKernelParam(128, 128, 64, 64, 64, flt_size, 1, 128, 32768, 1, 1, mma_shape);
                nominees.push_back(std::make_pair(nominee, 0.f));
            }
        }
    }

    std::sort(nominees.begin(), nominees.end(), SortByDescendScore);

    int declare_times        = 0;

    auto mgr = CodeGeneFactorManager::Instance();
    auto gene_factor = mgr->FindKernel(type);

    for(size_t i = 0; i < Min(32, nominees.size()); i++) {
        std::string source = "";
        auto& nominee = nominees[i].first;

        // printf("No.%d nominee %s : score %.2f \n", i, nominee.algo_name.c_str(), nominees[i].second);

        if (nominee.conv_type == "idxn") {
            gene_factor->GeneIdxnKernel(source, nominee.algo_name, nominee.mma_shape, nominee.tiles.flt_size, nominee.tiles.m_cta, nominee.tiles.n_cta, nominee.tiles.m_warp, nominee.tiles.n_warp, nominee.tiles.k_cta, nominee.tiles.k_per_step, declare_times);
            declare_times++;
        } else if (nominee.conv_type == "2spk") {
            gene_factor->Gene2spkKernel(source, nominee.algo_name, nominee.mma_shape, nominee.tiles.flt_size, nominee.tiles.m_cta, nominee.tiles.n_cta, nominee.tiles.m_warp, nominee.tiles.n_warp, nominee.tiles.k_cta, nominee.tiles.k_per_set, nominee.splitk, nominee.splitf, nominee.tiles.buf, declare_times);
            declare_times++;
        } else if (nominee.conv_type == "swzl") {
            gene_factor->GeneSwzlKernel(source, nominee.algo_name, nominee.mma_shape, nominee.tiles.flt_size, nominee.tiles.m_cta, nominee.tiles.n_cta, nominee.tiles.m_warp, nominee.tiles.n_warp, nominee.tiles.k_cta, nominee.splitk, nominee.tiles.buf, declare_times);
            declare_times++;
        }

        // printf("source is %s\n", source.c_str());

        // if (std::find(knames.begin(), knames.end(), algo_param.algo_name) == knames.end()) {
        //     sources = sources + source;
        // }

        sources = sources + source;

        knames.push_back(nominee.algo_name);
        params.push_back(nominee);

    }
#endif

    return ppl::common::RC_SUCCESS;
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
    double elapsed = 0.0f;
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::vector<std::string> knames;
    std::vector<algo_param_t> params;
    std::string sources = "";

    GetInt8ConvKernelNominees(device_id, type, conv_param, knames, params, sources);

    int index = 0;
    std::vector<const char *> compile_params;
    elapsed = AlgoForwardTimeInt8(device_id, stream, knames, sources, index, compile_params, device_id, true, type, d_input, d_flt, d_output, bias, d_temp_buf, params, conv_param, quant_param, fuse_param, workspace);

    algo_param = params[index];
#endif
    return elapsed;
}

void PPLCUDAConvolutionForwardJitImpInt8(
    int device_id,
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
#ifdef PPLNN_ENABLE_CUDA_JIT
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
    
    int tile_n = algo_param.tiles.n_cta;
    int tile_m = algo_param.tiles.m_cta;
    int cta_k  = algo_param.tiles.k_cta;

    dim3 block_size, grid_size;
    block_size.x = algo_param.tiles.cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    int smem_size = algo_param.tiles.smem_size;

    if(algo_param.conv_type == "swzl") {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_n);
        grid_size.y = DivUp(num_flt_per_grp_pad, tile_m);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_m);
        grid_size.y = DivUp(num_flt_per_grp_pad, tile_n);
    }
    grid_size.z = conv_param.num_grp * splitk * splitf * algo_param.gemm_batch;

    const int4 *pre_data  = (const int4 *)fuse_param.pre_data;
    const void *prelu     = (const void *)fuse_param.prelu;
    const void *elt_prelu = (const void *)fuse_param.elt_prelu;

    if(algo_param.conv_type == "idxn") {
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
    } else if (algo_param.conv_type == "2spk") {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        if(smem_size > MAX_STATIC_SMEM_SIZE_PER_CTA)
            cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);
        if (splitk == 1) {
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &quant_param.in_scale, &quant_param.d_flt_scale, &quant_param.out_scale, &quant_param.pre_scale, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;
            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, cta_k, splitk);
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &num_chl_per_spk_head, &num_chl_per_spk_tail, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &quant_param.in_scale, &quant_param.d_flt_scale};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
        }
    } else if (algo_param.conv_type == "swzl") {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        if(smem_size > MAX_STATIC_SMEM_SIZE_PER_CTA)
            cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);

        void *args[] = {&d_flt, &pad_input, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &quant_param.in_scale, &quant_param.d_flt_scale, &quant_param.out_scale, &quant_param.pre_scale, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
        CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
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
#endif
}
