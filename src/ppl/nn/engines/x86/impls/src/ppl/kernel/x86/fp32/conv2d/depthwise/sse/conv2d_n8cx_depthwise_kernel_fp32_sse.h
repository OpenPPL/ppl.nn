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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_SSE_CONV2D_N8CX_DEPTHWISE_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_SSE_CONV2D_N8CX_DEPTHWISE_KERNEL_FP32_SSE_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

#define KERNEL_FLAG_SUM()     (1 << 0)
#define KERNEL_FLAG_RELU()    (1 << 1)
#define KERNEL_FLAG_RELU6()   (1 << 2)

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define PRIV_PARAM_LEN() 10
#define SRC_IDX()        0
#define SUM_SRC_IDX()    1
#define DST_IDX()        2
#define FLT_IDX()        3
#define BIAS_IDX()       4
#define OW_IDX()         5
#define KH_START_IDX()   6
#define KH_END_IDX()     7
#define KW_START_IDX()   8
#define KW_END_IDX()     9

#define SHAR_PARAM_LEN()     6
#define SRC_SW_STRIDE_IDX()  0
#define SRC_DH_STRIDE_IDX()  1
#define SRC_DW_STRIDE_IDX()  2
#define FLAGS_IDX()          3
#define KW_IDX()             4

#define CH_DT_BLK() 8
#define CH_RF_BLK() 4

#define NT_STORE_OPT() 2
#define STRIDE_W_OPT() 3

#define MAX_OW_RF() 7

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n8cx_depthwise_kernel_fp32_sse_func_t)(const int64_t*, int64_t*);

extern conv2d_n8cx_depthwise_kernel_fp32_sse_func_t
    conv2d_n8cx_depthwise_kernel_fp32_sse_pad_table[NT_STORE_OPT()];
extern conv2d_n8cx_depthwise_kernel_fp32_sse_func_t
    conv2d_n8cx_depthwise_kernel_fp32_sse_blk_table[NT_STORE_OPT()][STRIDE_W_OPT()][MAX_OW_RF()];

}}}; // namespace ppl::kernel::x86

#endif
