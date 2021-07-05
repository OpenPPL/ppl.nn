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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_FMA_CONV2D_N16CX_WINOGRAD_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_FMA_CONV2D_N16CX_WINOGRAD_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define CH_DT_BLK()   16
#define CH_RF_BLK()   8

#define TILE_RF_CNT() 6
#define OC_RF_CNT()   2

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n16cx_winograd_kernel_fp32_fma_func_t)(
    const float *,
    const float *,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    float *);

extern conv2d_n16cx_winograd_kernel_fp32_fma_func_t
    conv2d_n16cx_winograd_kernel_fp32_fma_table[TILE_RF_CNT()];

}}}; // namespace ppl::kernel::x86

#endif
