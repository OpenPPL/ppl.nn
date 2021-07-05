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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_AVX512_CONV2D_IM2COL_GEMM_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_AVX512_CONV2D_IM2COL_GEMM_KERNEL_FP32_AVX512_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define KERNEL_FLAG_LD_BIAS() (1 << 0)
#define KERNEL_FLAG_RELU()    (1 << 1)
#define KERNEL_FLAG_RELU6()   (1 << 2)

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define KERNEL_PARAM_LEN()   12
#define SRC_IDX()            0
#define DST_IDX()            1
#define FLT_IDX()            2
#define BIAS_IDX()           3
#define OC_IDX()             4
#define K_IDX()              5
#define FLAGS_IDX()          6
#define SRC_HWB_STRIDE_IDX() 7
#define DST_HWB_STRIDE_IDX() 8
#define FLT_KB_STRIDE_IDX()  9
#define SIX_IDX()            10

#define NK_DT_BLK() 16

#define O6_HW_RF() 4
#define O6_OC_RF() 6

#define O9_HW_RF() 3
#define O9_OC_RF() 9

#define O14_HW_RF() 2
#define O14_OC_RF() 14

#define O31_HW_RF() 1
#define O31_OC_RF() 31

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_im2col_gemm_kernel_fp32_avx512_func_t)(const int64_t *);

extern conv2d_im2col_gemm_kernel_fp32_avx512_func_t
    conv2d_im2col_gemm_kernel_fp32_avx512_hw16_table[O14_OC_RF()];

extern conv2d_im2col_gemm_kernel_fp32_avx512_func_t
    conv2d_im2col_gemm_kernel_fp32_avx512_hw32_table[O14_OC_RF()];

extern conv2d_im2col_gemm_kernel_fp32_avx512_func_t
    conv2d_im2col_gemm_kernel_fp32_avx512_hw48_table[O9_OC_RF()];

extern conv2d_im2col_gemm_kernel_fp32_avx512_func_t
    conv2d_im2col_gemm_kernel_fp32_avx512_hw64_table[O6_OC_RF()];

}}}; // namespace ppl::kernel::x86

#endif
