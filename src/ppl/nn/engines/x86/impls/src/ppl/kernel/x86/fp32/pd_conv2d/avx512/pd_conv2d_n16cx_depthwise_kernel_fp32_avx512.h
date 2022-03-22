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

#ifndef __ST_PPL_KERNEL_X86_FP32_PD_CONV2D_AVX512_PD_CONV2D_N16CX_DEPTHWISE_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_PD_CONV2D_AVX512_PD_CONV2D_N16CX_DEPTHWISE_KERNEL_FP32_AVX512_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace kernel { namespace x86 {

class pd_conv2d_n16cx_depthwise_kernel_fp32_avx512 {
public:
    typedef void (*func_t)(int64_t*);

    struct param_def {
        static const int64_t SRC_PTR_KH_LIST_IDX = 0;
        static const int64_t DST_PTR_IDX = 1;
        static const int64_t FLT_PTR_IDX = 2;
        static const int64_t BIAS_PTR_IDX = 3;
        static const int64_t DST_WIDTH_IDX = 4;
        static const int64_t SRC_SW_STRIDE_IDX = 5;
        static const int64_t KH_START_IDX = 6;
        static const int64_t KH_END_IDX = 7;
        static const int64_t KW_IDX = 8;
        static const int64_t FLAGS_IDX = 9;
        static const int64_t LENGTH = 10;
    };

    struct config {
        static const int64_t CH_DATA_BLK = 16;
        static const int64_t MAX_W_REGS = 14;
        static const int64_t W_REG_ELTS = 1;
        static const int64_t CH_REGS = 1;
        static const int64_t CH_REG_ELTS = 16;
        static const int64_t CH_DATA_BLK_REGS = 1;
        static const int64_t CH_DATA_BLKS = CH_REGS / CH_DATA_BLK_REGS;
        static const int64_t MAX_W_BLK = MAX_W_REGS * W_REG_ELTS;
        static const int64_t CH_BLK = CH_DATA_BLKS * CH_DATA_BLK;
        static const int64_t NT_STORE_OPT = 2;
        static const int64_t SPEC_STRIDE_W_OPT = 3;
    };

    typedef int64_t flag_t;
    struct flag {
        static const flag_t RELU = (1 << 11);
        static const flag_t RELU6 = (1 << 12);
    };

    pd_conv2d_n16cx_depthwise_kernel_fp32_avx512(int64_t *param) : param_(param) { }
    inline void set_param(int64_t *param) { this->param_ = param; }
    inline int64_t *param() { return param_; }

    inline void execute(const int64_t nt_store, const int64_t spec_stride_w, const int64_t w_reg) {
        table_[nt_store][spec_stride_w][w_reg - 1](param_);
    }

private:
    int64_t *param_;
    static const func_t table_[config::NT_STORE_OPT][config::SPEC_STRIDE_W_OPT][config::MAX_W_REGS];
};

}}}; // namespace ppl::kernel::x86

#endif