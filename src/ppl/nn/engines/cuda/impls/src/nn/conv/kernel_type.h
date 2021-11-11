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

#ifndef __PPLCUDA_CONV_KERNEL_TYPE_H__
#define __PPLCUDA_CONV_KERNEL_TYPE_H__

#include <cuda.h>
#include <cuda_fp16.h>

#include "common/init_lut.h"

typedef void lut_kernel_t(
    int4* dA,
    int4* dB,
    int4* dC,
    int kloop_num,
    struct lut_t in_lut,
    int in_lut_size,
    struct lut_t flt_lut,
    int flt_lut_size,
    int in_hw,
    int out_hw,
    int flt_hw,
    int splitk,
    int in_height,
    int in_width,
    int in_num,
    int num_grp,
    int num_chl_per_grp,
    int num_chl_per_grp_pad,
    int flt_height,
    int flt_width,
    int num_flt_per_grp,
    int num_flt_per_grp_pad,
    int out_height,
    int out_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int hole_height,
    int hole_width,
    int has_bias,
    const int4* bias,
    int has_relu,
    const __half2 clip_min,
    bool has_clip,
    const __half2 clip_max,
    int has_prelu,
    const void* prelu,
    bool has_elt,
    const int4* pre_data,
    int has_elt_relu,
    const __half2 elt_clip_min,
    bool has_elt_clip,
    const __half2 elt_clip_max,
    int has_elt_prelu,
    const void* elt_prelu,
    const __half leaky,
    const __half elt_leaky,
    bool has_concat,
    int concat_offset_v8,
    int concat_stride_v8);

typedef void spk_kernel_t(
    int4* dA,
    int4* dB,
    int4* dC,
    int kloop_num,
    struct lut_t in_lut,
    int in_lut_size,
    struct lut_t flt_lut,
    int flt_lut_size,
    int num_chl_per_spk_head,
    int num_chl_per_spk_tail,
    int in_hw,
    int out_hw,
    int flt_hw,
    int splitk,
    int in_height,
    int in_width,
    int in_num,
    int num_grp,
    int num_chl_per_grp,
    int num_chl_per_grp_pad,
    int flt_height,
    int flt_width,
    int num_flt_per_grp,
    int num_flt_per_grp_pad,
    int out_height,
    int out_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int hole_height,
    int hole_width,
    int has_bias,
    int* bias);

typedef void idx_kernel_t(
    int4* dA,
    int4* dB,
    int4* dC,
    int kloop_num,
    int koff_num_pad,
    int in_hw,
    int out_hw,
    int flt_hw,
    int out_nhw,
    int in_height,
    int in_width,
    int in_num,
    int num_grp,
    int num_chl,
    int num_chl_per_grp,
    int in_chl_per_grp_pad,
    int flt_chl_per_grp_pad,
    int flt_height,
    int flt_width,
    int num_flt_per_grp,
    int num_flt_per_grp_pad,
    int out_height,
    int out_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int hole_height,
    int hole_width,
    int has_bias,
    const int4* bias,
    int has_relu,
    const __half2 clip_min,
    bool has_clip,
    const __half2 clip_max,
    int has_prelu,
    const void* prelu,
    bool has_elt,
    const int4* pre_data,
    int has_elt_relu,
    const __half2 elt_clip_min,
    bool has_elt_clip,
    const __half2 elt_clip_max,
    int has_elt_prelu,
    const void* elt_prelu,
    const __half leaky,
    const __half elt_leaky,
    bool has_concat,
    int concat_offset_v8,
    int concat_stride_v8);

#endif
