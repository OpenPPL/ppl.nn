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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_SSE_CONV2D_N8CX_GEMM_DIRECT_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_SSE_CONV2D_N8CX_GEMM_DIRECT_KERNEL_FP32_SSE_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define KERNEL_FLAG_LD_BIAS() (1 << 0)
#define KERNEL_FLAG_AD_BIAS() (1 << 1)
#define KERNEL_FLAG_RELU()    (1 << 2)
#define KERNEL_FLAG_RELU6()   (1 << 3)

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define PRIV_PARAM_LEN() 6
#define SRC_IDX()        0
#define HIS_IDX()        1
#define DST_IDX()        2
#define FLT_IDX()        3
#define BIAS_IDX()       4
#define HW_IDX()         5

#define SHAR_PARAM_LEN()     6
#define CHANNELS_IDX()       0
#define SRC_ICB_STRIDE_IDX() 1
#define HIS_OCB_STRIDE_IDX() 2
#define DST_OCB_STRIDE_IDX() 3
#define FLT_OCB_STRIDE_IDX() 4
#define FLAGS_IDX()          5

#define CH_DT_BLK() 8
#define CH_RF_BLK() 4

#define NT_STORE_OPT() 2

#define MAX_OC_RF() 14
#define MAX_HW_RF() 3

#define BLK1X3_OC_RF() 4
#define BLK1X3_HW_RF() 3

#define BLK1X1_OC_RF() 12
#define BLK1X1_HW_RF() 1

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n8cx_gemm_direct_kernel_fp32_sse_func_t)(const int64_t*, const int64_t*);

extern conv2d_n8cx_gemm_direct_kernel_fp32_sse_func_t
    conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw3_table[NT_STORE_OPT()][BLK1X3_OC_RF() / 2];
extern conv2d_n8cx_gemm_direct_kernel_fp32_sse_func_t
    conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw1_table[NT_STORE_OPT()][BLK1X1_OC_RF() / 2];

}}}; // namespace ppl::kernel::x86

#endif
