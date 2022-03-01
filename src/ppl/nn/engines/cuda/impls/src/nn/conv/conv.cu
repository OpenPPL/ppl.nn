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

#define SPK_KPARAM_LIST                                    \
    pad_input,                                             \
        d_flt,                                             \
        conv_out,                                          \
        kloop_num,                                         \
        in_lut, in_lut_size,                               \
        flt_lut, flt_lut_size,                             \
        num_chl_per_spk_head, num_chl_per_spk_tail,        \
        in_hw, out_hw,                                     \
        flt_hw, splitk,                                    \
        conv_param.in_height, conv_param.in_width,         \
        conv_param.in_num, conv_param.num_grp,             \
        num_chl_per_grp, num_chl_per_grp_pad,              \
        conv_param.flt_height, conv_param.flt_width,       \
        num_flt_per_grp, num_flt_per_grp_pad,              \
        conv_param.out_height, conv_param.out_width,       \
        conv_param.stride_height, conv_param.stride_width, \
        conv_param.pad_height, conv_param.pad_width,       \
        conv_param.hole_height, conv_param.hole_width,     \
        conv_param.has_bias, (int *)bias

#define LUT_KPARAM_LIST                                               \
    pad_input,                                                        \
        d_flt,                                                        \
        conv_out,                                                     \
        kloop_num,                                                    \
        in_lut, in_lut_size,                                          \
        flt_lut, flt_lut_size,                                        \
        in_hw, out_hw,                                                \
        flt_hw, splitk,                                               \
        conv_param.in_height, conv_param.in_width,                    \
        conv_param.in_num, conv_param.num_grp,                        \
        num_chl_per_grp, num_chl_per_grp_pad,                         \
        conv_param.flt_height, conv_param.flt_width,                  \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        conv_param.out_height, conv_param.out_width,                  \
        conv_param.stride_height, conv_param.stride_width,            \
        conv_param.pad_height, conv_param.pad_width,                  \
        conv_param.hole_height, conv_param.hole_width,                \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

#define SWZL_SPK_KPARAM_LIST                               \
        d_flt,                                             \
        pad_input,                                         \
        conv_out,                                          \
        kloop_num,                                         \
        in_lut, in_lut_size,                               \
        flt_lut, flt_lut_size,                             \
        num_chl_per_spk_head, num_chl_per_spk_tail,        \
        in_hw, out_hw,                                     \
        flt_hw, splitk,                                    \
        conv_param.in_height, conv_param.in_width,         \
        conv_param.in_num, conv_param.num_grp,             \
        num_chl_per_grp, num_chl_per_grp_pad,              \
        conv_param.flt_height, conv_param.flt_width,       \
        num_flt_per_grp, num_flt_per_grp_pad,              \
        conv_param.out_height, conv_param.out_width,       \
        conv_param.stride_height, conv_param.stride_width, \
        conv_param.pad_height, conv_param.pad_width,       \
        conv_param.hole_height, conv_param.hole_width,     \
        conv_param.has_bias, (int *)bias

#define SWZL_LUT_KPARAM_LIST                                          \
        d_flt,                                                        \
        pad_input,                                                    \
        conv_out,                                                     \
        kloop_num,                                                    \
        in_lut, in_lut_size,                                          \
        flt_lut, flt_lut_size,                                        \
        in_hw, out_hw,                                                \
        flt_hw, splitk,                                               \
        conv_param.in_height, conv_param.in_width,                    \
        conv_param.in_num, conv_param.num_grp,                        \
        num_chl_per_grp, num_chl_per_grp_pad,                         \
        conv_param.flt_height, conv_param.flt_width,                  \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        conv_param.out_height, conv_param.out_width,                  \
        conv_param.stride_height, conv_param.stride_width,            \
        conv_param.pad_height, conv_param.pad_width,                  \
        conv_param.hole_height, conv_param.hole_width,                \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

#define IDX_KPARAM_LIST                                               \
    pad_input,                                                        \
        d_flt,                                                        \
        conv_out,                                                     \
        kloop_num, koff_num_pad,                                      \
        in_hw, out_hw,                                                \
        flt_hw, out_nhw,                                              \
        conv_param.in_height, conv_param.in_width,                    \
        conv_param.in_num, conv_param.num_grp,                        \
        conv_param.num_chl, num_chl_per_grp,                          \
        in_chl_per_grp_pad, flt_chl_per_grp_pad,                      \
        conv_param.flt_height, conv_param.flt_width,                  \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        conv_param.out_height, conv_param.out_width,                  \
        conv_param.stride_height, conv_param.stride_width,            \
        conv_param.pad_height, conv_param.pad_width,                  \
        conv_param.hole_height, conv_param.hole_width,                \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

#define MERGE_KPARAM_LIST                                             \
    conv_out, final_out,                                              \
        spk_height_v1, spk_width_v8,                                  \
        out_hw, splitk *splitf,                                       \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

static std::vector<kernel_info_t> g_kernel_container;
static std::vector<kernel_info_t> g_int8_kernel_container;
static bool is_g_kernel_container_initialized = false;

static std::unordered_map<size_t, algo_param_t> g_conv_shape_hash;

__inline__ void InitializeKernelContainerInt8(std::vector<kernel_info_t> &g_kernel_container, ppl::common::datatype_t type)
{
    if (type == ppl::common::DATATYPE_FLOAT16) {
#ifndef PPLNN_ENABLE_CUDA_JIT
        Initialize2spkConvF1KernelContainer(g_kernel_container);
        Initialize2spkConvF3KernelContainer(g_kernel_container);
        Initialize2spkConvFNKernelContainer(g_kernel_container);
        Initialize2spkConvFSKernelContainer(g_kernel_container);

        InitializeIdxnConvKernelContainer(g_kernel_container);

        InitializeSwzlConvF1KernelContainer(g_kernel_container);
        InitializeSwzlConvF3KernelContainer(g_kernel_container);
        InitializeSwzlConvFNKernelContainer(g_kernel_container);
#endif
    }

    is_g_kernel_container_initialized = true;
}

std::string GetConvShapeString(const conv_param_t &conv_param)
{
    return std::string("b" + std::to_string(conv_param.in_num) +
                       "_c" + std::to_string(conv_param.num_chl) +
                       "_d" + std::to_string(conv_param.num_flt) +
                       "_g" + std::to_string(conv_param.num_grp) +
                       "_h" + std::to_string(conv_param.in_height) +
                       "_w" + std::to_string(conv_param.in_width) +
                       "_r" + std::to_string(conv_param.flt_height) +
                       "_s" + std::to_string(conv_param.flt_width) +
                       "_p" + std::to_string(conv_param.pad_height) +
                       "_q" + std::to_string(conv_param.pad_width) +
                       "_u" + std::to_string(conv_param.stride_height) +
                       "_v" + std::to_string(conv_param.stride_width) +
                       "_y" + std::to_string(conv_param.hole_height) +
                       "_x" + std::to_string(conv_param.hole_width) +
                       "_");
}

__inline__ size_t GetConvShapeHashKey(conv_param_t &conv_param)
{
    return std::hash<std::string>{}(GetConvShapeString(conv_param));
}

uint64_t PPLCUDAConvolutionGetCompilationBufSize(ppl::common::datatype_t type, conv_param_t &conv_param, uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t cvt_input_size = 0;
    uint64_t cvt_output_size = 0;

    if (is_in_grp_pad)
        cvt_input_size = GetCvtInputSize(type, conv_param, num_chl_per_grp_pad);

    if (is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t split_size = GetMaxSplitSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t total_size = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}
uint64_t PPLCUDAConvolutionGetRuntimeBufSize(
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    unsigned int splitk,
    unsigned int splitf,
    uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t cvt_input_size = 0;
    uint64_t cvt_output_size = 0;

    if (is_in_grp_pad)
        cvt_input_size = GetCvtInputSize(type, conv_param, num_chl_per_grp_pad);
    if (is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t split_size = 0;
    
    if(splitk > 1 || splitf > 1)
        split_size = GetSplitKFSize(type, conv_param, num_flt_per_grp_pad, splitk, splitf);

    uint64_t total_size = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}

/* -----------------  FP16 KERNEL ------------------ */

double PPLCUDAConvolutionSelectKernel(
    cudaStream_t &stream,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param,
    uint64_t workspace)
{
    if (!is_g_kernel_container_initialized)
        InitializeKernelContainerInt8(g_kernel_container, type);

    size_t conv_shape_hash = GetConvShapeHashKey(conv_param);

    std::unordered_map<size_t, algo_param_t>::const_iterator conv_shape_hash_iterator = g_conv_shape_hash.find(conv_shape_hash);

    if (conv_shape_hash_iterator != g_conv_shape_hash.end()) {
        algo_param.kid    = conv_shape_hash_iterator->second.kid;
        algo_param.splitk = conv_shape_hash_iterator->second.splitk;
        algo_param.splitf = conv_shape_hash_iterator->second.splitf;

        return ppl::common::RC_SUCCESS;
    }

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

    int4 *final_out = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half leaky         = __float2half(fuse_param.leaky);
    __half elt_leaky     = __float2half(fuse_param.elt_leaky);

    float minTime = FLT_MAX;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    const int SPLITK_OPTIONS[] = {1, 2, 4, 8};

    for (unsigned int spk = 0; spk < 4; spk++) {
        unsigned int splitk = SPLITK_OPTIONS[spk];

        for (unsigned int kid = 0; kid < g_kernel_container.size(); kid++) {
            unsigned int splitf = (g_kernel_container[kid].ktype == CONV_2SPK_FS) ? flt_hw : 1;

            if (!g_kernel_container[kid].CheckKernelTypeFeasible(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk))
                continue;

            if (!g_kernel_container[kid].CheckSplitkFeasible(num_chl_per_grp, splitk))
                continue;

            if (!g_kernel_container[kid].CheckSplitfFeasible(splitf, splitk))
                continue;

            int4 *conv_out = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

            dim3 block_size, grid_size;

            block_size.x = g_kernel_container[kid].cta_size_in_thd;
            block_size.y = 1;
            block_size.z = 1;

            if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
                    g_kernel_container[kid].ktype == CONV_SWZL_FN) {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_n_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_m_per_cta);
            } else {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_m_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_n_per_cta);
            }

            grid_size.z = conv_param.num_grp * splitk * splitf;

            cudaEventRecord(begin, stream);

            for (int i = 0; i < TIMES; i++) {
                if (g_kernel_container[kid].ktype == CONV_IDXN_C2 || g_kernel_container[kid].ktype == CONV_IDXN_C4 ||
                    g_kernel_container[kid].ktype == CONV_IDXN_C32) {
                    int tile_k_per_step = g_kernel_container[kid].tile_k_per_step;

                    int img_pad_size = pad_size;
                    int flt_pad_size = g_kernel_container[kid].flt_pad_size;
                    int out_nhw      = out_hw * conv_param.in_num;

                    int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
                    int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
                    int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

                    int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);
                    int koff_num_pad = Align(kloop_num * (g_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

                    (g_kernel_container[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);
                } else if (g_kernel_container[kid].ktype == CONV_2SPK_F1 || g_kernel_container[kid].ktype == CONV_2SPK_F3 ||
                           g_kernel_container[kid].ktype == CONV_2SPK_FN || g_kernel_container[kid].ktype == CONV_2SPK_FS ||
                           g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 ||
                           g_kernel_container[kid].ktype == CONV_SWZL_FN) {

                    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);

                    lut_t in_lut, flt_lut;
                    int in_lut_size, flt_lut_size;

                    InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, g_kernel_container[kid].tile_k_per_cta, pad_size);

                    InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta, pad_size);

                    if (splitk == 1) {
                        if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || g_kernel_container[kid].ktype == CONV_SWZL_FN)
                            (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_LUT_KPARAM_LIST);
                        else {
                            (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(LUT_KPARAM_LIST);

                        }
                    } else {
                        int num_chl_per_spk_head, num_chl_per_spk_tail;

                        InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_kernel_container[kid].tile_k_per_cta, splitk);

                        if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || g_kernel_container[kid].ktype == CONV_SWZL_FN)
                            (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_SPK_KPARAM_LIST);
                        else
                            (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SPK_KPARAM_LIST);
                    }

                    if (splitk > 1 || splitf > 1) {
                        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
                        int spk_height_v1 = out_hw * conv_param.in_num;

                        dim3 merge_grid_size, merge_block_size;
                        merge_block_size.x = 64; // empirical value
                        merge_block_size.y = 1;
                        merge_block_size.z = 1;

                        merge_grid_size.x = spk_height_v1;
                        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
                        merge_grid_size.z = 1;

                        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
                    }
                }
            }

            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed, begin, end);

            if (elapsed < minTime) {
                algo_param.kid    = kid;
                algo_param.splitk = splitk;
                algo_param.splitf = splitf;
                minTime           = elapsed;
            }
        }
    }

    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    g_conv_shape_hash[conv_shape_hash] = algo_param;
    return minTime;
}

void PPLCUDAConvolutionForwardImp(
    cudaStream_t &stream,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param)
{
    if (!is_g_kernel_container_initialized)
        InitializeKernelContainerInt8(g_kernel_container, type);

    unsigned int kid    = algo_param.kid;
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

    int4 *final_out = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half leaky         = __float2half(fuse_param.leaky);
    __half elt_leaky     = __float2half(fuse_param.elt_leaky);

    dim3 block_size, grid_size;

    block_size.x = g_kernel_container[kid].cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || \
            g_kernel_container[kid].ktype == CONV_SWZL_FN) {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_n_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_m_per_cta);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_m_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_n_per_cta);
    }

    grid_size.z = conv_param.num_grp * splitk * splitf;

    if (g_kernel_container[kid].ktype == CONV_IDXN_C2 || g_kernel_container[kid].ktype == CONV_IDXN_C4 ||
        g_kernel_container[kid].ktype == CONV_IDXN_C32) {
        int img_pad_size = pad_size;
        int flt_pad_size = g_kernel_container[kid].flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

        int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);
        int koff_num_pad = Align(kloop_num * (g_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

        (g_kernel_container[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);

    } else if (g_kernel_container[kid].ktype == CONV_2SPK_F1 || g_kernel_container[kid].ktype == CONV_2SPK_F3 ||
               g_kernel_container[kid].ktype == CONV_2SPK_FN || g_kernel_container[kid].ktype == CONV_2SPK_FS ||
               g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 ||
               g_kernel_container[kid].ktype == CONV_SWZL_FN) {

        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, g_kernel_container[kid].tile_k_per_cta, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta, pad_size);

        if (splitk == 1) {
            if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || g_kernel_container[kid].ktype == CONV_SWZL_FN)
                (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_LUT_KPARAM_LIST);
            else {
                (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(LUT_KPARAM_LIST);
            }

        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;

            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_kernel_container[kid].tile_k_per_cta, splitk);


            if(g_kernel_container[kid].ktype == CONV_SWZL_F1 || g_kernel_container[kid].ktype == CONV_SWZL_F3 || g_kernel_container[kid].ktype == CONV_SWZL_FN)
                (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SWZL_SPK_KPARAM_LIST);
            else
                (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SPK_KPARAM_LIST);
        }
    }

    if (splitk > 1 || splitf > 1) {
        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1 = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x = spk_height_v1;
        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }

    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
}

/* -----------------  JIT FP16 KERNEL ------------------ */

#define MAX_KERNEL_SIZE (1 + 12 + 30)

__inline__ std::string ToString(int v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

ppl::common::RetCode PPLCUDAConvolutionLoadAlgoParam(
    algo_param_t &algo_param)
{
    auto kname        = algo_param.algo_name.substr(algo_param.algo_name.find("_b"));
    auto f_size       = algo_param.algo_name.substr(25,2);

    if (algo_param.algo_name.find("Idxn") != std::string::npos) {
        sscanf(kname.c_str(), "_b%dx%d_w%dx%d_k%d_s%d_nosmem", &algo_param.tiles.m_cta, &algo_param.tiles.n_cta, &algo_param.tiles.m_warp, &algo_param.tiles.n_warp, &algo_param.tiles.k_cta, &algo_param.tiles.k_per_step);
        algo_param.tiles.flt_pad_size    = algo_param.tiles.k_per_step / 4;
        algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                           (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                           WARP_SIZE;
    } else if (algo_param.algo_name.find("2spk") != std::string::npos) { // Use 2spk algo for large channel
        sscanf(kname.c_str(), "_b%dx%d_w%dx%d_k%d_s%d_buf%d", &algo_param.tiles.m_cta, &algo_param.tiles.n_cta, &algo_param.tiles.m_warp, &algo_param.tiles.n_warp, &algo_param.tiles.k_cta, &algo_param.tiles.k_per_set, &algo_param.tiles.buf);
        if (f_size == "f1" || f_size == "fs") {
            algo_param.tiles.flt_size = 1;
        } else if (f_size == "f3") {
            algo_param.tiles.flt_size = 3;
        } else if (f_size == "fn") {
            algo_param.tiles.flt_size = 0;
        } else {
            return ppl::common::RC_INVALID_VALUE;
        }
        algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                           (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                           (algo_param.tiles.k_cta / algo_param.tiles.k_per_set) *
                                           WARP_SIZE;
        if (algo_param.splitk > 1)
            algo_param.algo_name = algo_param.algo_name + "_splitk";
    } else if (algo_param.algo_name.find("Swzl") != std::string::npos) {
        sscanf(kname.c_str(), "_b%dx%d_w%dx%d_k%d_buf%d", &algo_param.tiles.m_cta, &algo_param.tiles.n_cta, &algo_param.tiles.m_warp, &algo_param.tiles.n_warp, &algo_param.tiles.k_cta, &algo_param.tiles.buf);
        if (f_size == "f1") {
            algo_param.tiles.flt_size = 1;
        } else if (f_size == "f3") {
            algo_param.tiles.flt_size = 3;
        } else if (f_size == "fn") {
            algo_param.tiles.flt_size = 0;
        } else {
            return ppl::common::RC_INVALID_VALUE;
        }
        algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                           (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                           WARP_SIZE;
        if (algo_param.splitk > 1)
            algo_param.algo_name = algo_param.algo_name + "_splitk";
    } else {
        return ppl::common::RC_NOT_FOUND;
    }
    return ppl::common::RC_SUCCESS;
}

void ModifySingleParam(algo_param_t &algo_param, int pos, int offset)
{
    switch (pos) {
        case 0:
            algo_param.tiles.m_cta = offset > 0 ? algo_param.tiles.m_cta * 2 : algo_param.tiles.m_cta / 2;
            break;
        case 1:
            algo_param.tiles.n_cta = offset > 0 ? algo_param.tiles.n_cta * 2 : algo_param.tiles.n_cta / 2;
            break;
        case 2:
            algo_param.tiles.m_warp = offset > 0 ? algo_param.tiles.m_warp * 2 : algo_param.tiles.m_warp / 2;
            break;
        case 3:
            algo_param.tiles.n_warp = offset > 0 ? algo_param.tiles.n_warp * 2 : algo_param.tiles.n_warp / 2;
            break;
        case 4:
            algo_param.tiles.k_cta = offset > 0 ? algo_param.tiles.k_cta * 2 : algo_param.tiles.k_cta / 2;
            break;
        case 5:
            algo_param.tiles.k_per_step = offset > 0 ? algo_param.tiles.k_per_step * 2 : algo_param.tiles.k_per_step / 2;
            algo_param.tiles.k_per_set  = offset > 0 ? algo_param.tiles.k_per_set * 2 : algo_param.tiles.k_per_set / 2;
            break;
    }
}

ppl::common::RetCode PPLCUDAConvolutionModifyAlgoParam(
    algo_param_t &algo_param,
    uint32_t index)
{
    if (index == 0) {
        return ppl::common::RC_SUCCESS;
    } else if (index <= 12) {
        index      = index - 1;
        int pos    = index / 6;
        int offset = index % 2;
        ModifySingleParam(algo_param, pos, offset);
    } else {
        index     = index - 13;
        int pos_1 = index / 5;
        int pos_2 = index % 5;
        if (pos_2 >= pos_1)
            pos_2++;
        ModifySingleParam(algo_param, pos_1, 1);
        ModifySingleParam(algo_param, pos_2, 0);
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAConvolutionPredictKernel(
    ppl::common::datatype_t type,
    algo_param_t &algo_param,
    conv_param_t &conv_param)
{
    int in_hw       = conv_param.in_num * conv_param.in_height * conv_param.in_width;
    int out_hw      = conv_param.in_num * conv_param.out_height * conv_param.out_width;
    int flt_hw      = conv_param.flt_height * conv_param.flt_width;
    int chl_per_grp = conv_param.num_chl / conv_param.num_grp;

    if (chl_per_grp <= 32) { // Use non-shared memory algo for small channel
        if (flt_hw > 9) {
            algo_param.tiles.m_cta  = 128;
            algo_param.tiles.m_warp = 64;
        } else {
            algo_param.tiles.m_cta  = 32;
            algo_param.tiles.m_warp = 16;
        }

        if (in_hw == out_hw) {
            algo_param.tiles.n_cta  = 64;
            algo_param.tiles.n_warp = 32;
        } else {
            algo_param.tiles.n_cta  = 32;
            algo_param.tiles.n_warp = 16;
        }

        algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                           (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                           WARP_SIZE;

        if (chl_per_grp <= 2) {
            algo_param.tiles.k_cta      = 8;
            algo_param.tiles.k_per_step = 8;
        } else if (chl_per_grp <= 4) {
            algo_param.tiles.k_cta      = 16;
            algo_param.tiles.k_per_step = 16;
        } else {
            algo_param.tiles.k_cta      = 32;
            algo_param.tiles.k_per_step = 32;
        }
        if (type == ppl::common::DATATYPE_INT8) {
            algo_param.tiles.k_cta      *= 2;
            algo_param.tiles.k_per_step *= 2;            
        }
    } else { // Use 2spk or swizzle algo for large channel
        float min_pad          = 1.0;
        algo_param.tiles.m_cta = 16;
        for (int32_t i = 128; i >= 16; i = i / 2) {
            if (out_hw < i)
                continue;
            float pad = 1.0 * (DivUp(out_hw, i) * i - out_hw) / out_hw;
            if (pad < min_pad) {
                min_pad                = pad;
                algo_param.tiles.m_cta = i;
            }
            if (min_pad < 0.1)
                break;
        }

        algo_param.tiles.n_cta = 16;
        for (int32_t i = 128; i >= 16; i = i / 2) {
            int cout = conv_param.num_flt;
            if ((cout < 64 && i / cout == 1) || (cout >= 64 && cout % i == 0)) {
                algo_param.tiles.n_cta = i;
                break;
            }
        }

        if (conv_param.num_chl >= 128) {
            algo_param.tiles.k_cta = 64;
        } else {
            algo_param.tiles.k_cta = 32;
        }

        if (algo_param.tiles.m_cta == 128 && algo_param.tiles.n_cta == 128) {
            algo_param.tiles.m_cta = 64;
        }

        if (algo_param.tiles.m_cta * 4 < algo_param.tiles.n_cta) {
            algo_param.tiles.m_cta *= 2;
            algo_param.tiles.n_cta /= 2;
        }
        if (algo_param.tiles.n_cta * 4 < algo_param.tiles.m_cta) {
            algo_param.tiles.m_cta /= 2;
            algo_param.tiles.n_cta *= 2;
        }

        algo_param.tiles.m_warp    = algo_param.tiles.m_cta / 2;
        algo_param.tiles.n_warp    = algo_param.tiles.n_cta / 2;
        algo_param.tiles.k_per_set = algo_param.tiles.k_cta / 2;
        if (algo_param.tiles.k_per_set <= 8) {
            algo_param.tiles.k_per_set = 16;
        }
        if (algo_param.tiles.m_warp <= 8) {
            algo_param.tiles.m_warp = 16;
        }
        if (algo_param.tiles.n_warp <= 8) {
            algo_param.tiles.n_warp = 16;
        }
    }
    return ppl::common::RC_SUCCESS;
}

float AlgoForwardTime(
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
            PPLCUDAConvolutionForwardJitImp(
                stream, function, type, d_input, d_flt, d_output, bias, d_temp_buf, algo_param[n], conv_param, fuse_param);
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

double PPLCUDAConvolutionJitSelectKernel(
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
            unsigned int KERNEL_TYPE = spf ? 1 : 2;

            if (spf == 1 && flt_hw == 1)
                continue;
            if (spf >= 1 && spk >= 1)
                continue;
            if ((spf >= 1 || spk >= 1) && num_chl_per_grp <= 32)
                continue;

            for (unsigned int index = 0; index < MAX_KERNEL_SIZE * KERNEL_TYPE; index++) {
                conv_ktype_t ktype;
                algo_param = pre_algo_param;
                PPLCUDAConvolutionModifyAlgoParam(algo_param, index % MAX_KERNEL_SIZE); // change algo_param
                algo_param.splitk = splitk;
                algo_param.splitf = splitf;

                int size_x    = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, algo_param.tiles.m_cta);
                int size_y    = DivUp(num_flt_per_grp_pad, algo_param.tiles.n_cta);
                int grid_size = size_x * size_y * conv_param.num_grp;

                if (num_chl_per_grp <= 32) { // Use non-shared memory algo for small channel
                    algo_param.tiles.flt_pad_size = algo_param.tiles.k_per_step / 4;
                    if (algo_param.tiles.k_per_step <= 8) {
                        ktype = CONV_IDXN_C2;
                    } else if (algo_param.tiles.k_per_step <= 16) {
                        ktype = CONV_IDXN_C4;
                    } else {
                        ktype = CONV_IDXN_C32;
                    }
                    algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                                    (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                                    WARP_SIZE;
                    algo_param.algo_name = "nvIdxnConv_hmma1688_nhwc_b" + ToString(algo_param.tiles.m_cta) + "x" + ToString(algo_param.tiles.n_cta) +
                                        "_w" + ToString(algo_param.tiles.m_warp) + "x" + ToString(algo_param.tiles.n_warp) +
                                        "_k" + ToString(algo_param.tiles.k_cta) + "_s" + ToString(algo_param.tiles.k_per_step) + "_nosmem";
                } else if (index < MAX_KERNEL_SIZE) { // Use 2spk algo for large channel
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
                    algo_param.algo_name = "nv2spkConv_hmma1688_nhwc_" + f_size + "_b" + ToString(algo_param.tiles.m_cta) + "x" + ToString(algo_param.tiles.n_cta) +
                                        "_w" + ToString(algo_param.tiles.m_warp) + "x" + ToString(algo_param.tiles.n_warp) +
                                        "_k" + ToString(algo_param.tiles.k_cta) + "_s" + ToString(algo_param.tiles.k_per_set) + "_buf1";
                    if (splitk > 1) {
                        algo_param.algo_name = algo_param.algo_name + "_splitk";
                    }
                } else { // Use swzl algo for large channel
                    algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *
                                                    (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *
                                                    WARP_SIZE;
                    ktype                     = CONV_SWZL_FN;
                    std::string f_size        = "fn";
                    algo_param.tiles.flt_size = 0;
                    if (conv_param.flt_height == 1 && conv_param.flt_width == 1) {
                        ktype                     = CONV_SWZL_F1;
                        f_size                    = "f1";
                        algo_param.tiles.flt_size = 1;
                    } else if (conv_param.flt_height == 3 && conv_param.flt_width == 3) {
                        ktype                     = CONV_SWZL_F3;
                        f_size                    = "f3";
                        algo_param.tiles.flt_size = 3;
                    }
                    algo_param.algo_name = "nvSwzlConv_hmma1688_nhwc_" + f_size + "_b" + ToString(algo_param.tiles.m_cta) + "x" + ToString(algo_param.tiles.n_cta) +
                                        "_w" + ToString(algo_param.tiles.m_warp) + "x" + ToString(algo_param.tiles.n_warp) +
                                        "_k" + ToString(algo_param.tiles.k_cta) + "_buf1";
                    if (splitk > 1) {
                        algo_param.algo_name = algo_param.algo_name + "_splitk";
                    }
                }

                kernel_info_t temp_kernel(-1, ktype, algo_param.algo_name.c_str());
                if (!temp_kernel.CheckKernelTilesFeasible(type, device_id))
                    continue;
                if (!temp_kernel.CheckKernelTypeFeasible(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk))
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
                if (algo_param.algo_name.find("Idxn") != std::string::npos) {
                    gene_factor->GeneIdxnKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta, algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta, algo_param.tiles.k_per_step, declare_times);
                    declare_times++;
                } else if (algo_param.algo_name.find("2spk") != std::string::npos) {
                    gene_factor->Gene2spkKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta, algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta, algo_param.tiles.k_per_set, algo_param.splitk, algo_param.splitf, algo_param.tiles.buf, declare_times);
                    declare_times++;
                } else if (algo_param.algo_name.find("Swzl") != std::string::npos) {
                    gene_factor->GeneSwzlKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta, algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta, algo_param.splitk, algo_param.tiles.buf, declare_times);
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
    elapsed = AlgoForwardTime(stream, knames, total_source, index, compile_params, device_id, true, type, d_input, d_flt, d_output, bias, d_temp_buf, params, conv_param, fuse_param, workspace);

    algo_param                         = params[index];
    return elapsed;
}

void PPLCUDAConvolutionForwardJitImp(
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

    int4 *final_out = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half leaky         = __float2half(fuse_param.leaky);
    __half elt_leaky     = __float2half(fuse_param.elt_leaky);

    int tile_n = algo_param.tiles.n_cta;
    int tile_m = algo_param.tiles.m_cta;
    int cta_k  = algo_param.tiles.k_cta;

    dim3 block_size, grid_size;
    block_size.x = algo_param.tiles.cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    if(algo_param.algo_name.find("Swzl") != std::string::npos) {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_n);
        grid_size.y = DivUp(num_flt_per_grp_pad, tile_m);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_m);
        grid_size.y = DivUp(num_flt_per_grp_pad, tile_n);
    }
    grid_size.z = conv_param.num_grp * splitk * splitf;

    const int4 *pre_data  = (const int4 *)fuse_param.pre_data;
    const void *prelu     = (const void *)fuse_param.prelu;
    const void *elt_prelu = (const void *)fuse_param.elt_prelu;

    if (algo_param.algo_name.find("Idxn") != std::string::npos) {
        int img_pad_size = pad_size;
        int flt_pad_size = algo_param.tiles.flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

        int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, cta_k);
        int koff_num_pad = Align(kloop_num * (cta_k / flt_pad_size), WARP_SIZE);

        void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &koff_num_pad, &in_hw, &out_hw, &flt_hw, &out_nhw, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &conv_param.num_chl, &num_chl_per_grp, &in_chl_per_grp_pad, &flt_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};

        CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
    } else if (algo_param.algo_name.find("2spk") != std::string::npos) {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);
        if (splitk == 1) {
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;

            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, cta_k, splitk);

            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &num_chl_per_spk_head, &num_chl_per_spk_tail, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
        }
    } else if (algo_param.algo_name.find("Swzl") != std::string::npos) {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);
        if (splitk == 1) {
            void *args[] = {&d_flt, &pad_input, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;

            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, cta_k, splitk);
            void *args[] = {&d_flt, &pad_input, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &num_chl_per_spk_head, &num_chl_per_spk_tail, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
        }
    }

    if (splitk > 1 || splitf > 1) {
        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1 = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x = spk_height_v1;
        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }
    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
}
