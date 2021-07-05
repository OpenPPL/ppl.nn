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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_IMPLICIT_GEMM_FMA_CONV2D_N16CX_IMPLICIT_GEMM_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_IMPLICIT_GEMM_FMA_CONV2D_N16CX_IMPLICIT_GEMM_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define KERNEL_FLAG_LD_BIAS() (1 << 0)
#define KERNEL_FLAG_AD_BIAS() (1 << 1)
#define KERNEL_FLAG_RELU()    (1 << 2)
#define KERNEL_FLAG_RELU6()   (1 << 3)

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define PRIV_PARAM_LEN() 8
#define SRC_IDX()        0
#define HIS_IDX()        1
#define DST_IDX()        2
#define FLT_IDX()        3
#define BIAS_IDX()       4
#define OH_IDX()         5
#define OW_IDX()         6

#define SHAR_PARAM_LEN()     16
#define CHANNELS_IDX()       0
#define SRC_ICB_STRIDE_IDX() 1
#define SRC_SH_STRIDE_IDX()  2
#define SRC_SW_STRIDE_IDX()  3
#define SRC_DH_STRIDE_IDX()  4
#define SRC_DW_STRIDE_IDX()  5
#define HIS_H_STRIDE_IDX()   6
#define DST_H_STRIDE_IDX()   7
#define FLT_K_STRIDE_IDX()   8
#define FLAGS_IDX()          9
#define KH_IDX()             10
#define KW_IDX()             11

#define STRIDE_W_OPT() 3
#define NT_STORE_OPT() 2
#define PREF_SRC_OPT() 2

#define CH_DT_BLK() 16
#define OC_RF_BLK() 8

#define BLK1X6_OC_RF() 2
#define BLK1X6_OW_RF() 6

#define BLK1X3_OC_RF() 4
#define BLK1X3_OW_RF() 3

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n16cx_implicit_gemm_kernel_fp32_fma_func_t)(const int64_t*, const int64_t*);

extern conv2d_n16cx_implicit_gemm_kernel_fp32_fma_func_t
    conv2d_n16cx_implicit_gemm_kernel_fp32_fma_blk1x6_table[STRIDE_W_OPT()][NT_STORE_OPT()][PREF_SRC_OPT()][BLK1X6_OC_RF()][BLK1X6_OW_RF()];

}}}; // namespace ppl::kernel::x86

#endif
