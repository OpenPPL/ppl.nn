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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_AVX512_CONV2D_N16CX_DIRECT_NDARRAY_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_AVX512_CONV2D_N16CX_DIRECT_NDARRAY_KERNEL_FP32_AVX512_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

#define KERNEL_FLAG_LD_BIAS() (1 << 0)
#define KERNEL_FLAG_AD_BIAS() (1 << 1)
#define KERNEL_FLAG_RELU()    (1 << 2)
#define KERNEL_FLAG_RELU6()   (1 << 3)

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define PRIV_PARAM_LEN() 10
#define SRC_IDX()        0
#define HIS_IDX()        1
#define DST_IDX()        2
#define FLT_IDX()        3
#define BIAS_IDX()       4
#define OW_IDX()         5
#define KH_START_IDX()   6
#define KH_END_IDX()     7
#define KW_START_IDX()   8
#define KW_END_IDX()     9

#define SHAR_PARAM_LEN()     10
#define CHANNELS_IDX()       0
#define SRC_C_STRIDE_IDX()   1
#define SRC_H_STRIDE_IDX()   2
#define HIS_OCB_STRIDE_IDX() 3
#define DST_OCB_STRIDE_IDX() 4
#define FLT_OCB_STRIDE_IDX() 5
#define FLAGS_IDX()          6
#define SW_IDX()             7
#define KH_IDX()             8
#define KW_IDX()             9

#define OC_DT_BLK() 16

#define NT_STORE_OPT() 2

#define MAX_OC_RF() 2
#define MAX_OW_RF() 7

#define BLK1X14_OC_RF() 2
#define BLK1X14_OW_RF() 14

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t)(const int64_t*, const int64_t*);

extern conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t
    conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_pad_table[NT_STORE_OPT()][MAX_OC_RF()];
extern conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t
    conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_o16_table[NT_STORE_OPT()][MAX_OW_RF()];
extern conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t
    conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_o32_table[NT_STORE_OPT()][MAX_OW_RF()];

}}}; // namespace ppl::kernel::x86

#endif
