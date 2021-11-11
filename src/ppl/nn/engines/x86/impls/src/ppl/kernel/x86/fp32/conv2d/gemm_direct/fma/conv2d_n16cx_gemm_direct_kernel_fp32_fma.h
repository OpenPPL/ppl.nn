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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_FMA_CONV2D_N16CX_GEMM_DIRECT_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_FMA_CONV2D_N16CX_GEMM_DIRECT_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace kernel { namespace x86 {

class conv2d_n16cx_gemm_direct_kernel_fp32_fma {
public:
    typedef void (*func_t)(int64_t*);

    struct param_def {
        static const int64_t SRC_PTR_IDX = 0;
        static const int64_t HIS_PTR_IDX = 1;
        static const int64_t DST_PTR_IDX = 2;
        static const int64_t FLT_PTR_IDX = 3;
        static const int64_t BIAS_PTR_IDX = 4;
        static const int64_t SPACE_IDX = 5;
        static const int64_t CHANNELS_IDX = 6;
        static const int64_t SRC_ICB_STRIDE_IDX = 7;
        static const int64_t FLAGS_IDX = 8;
        static const int64_t LENGTH = 9;
    };

    struct config {
        static const int64_t IC_DATA_BLK = 16;
        static const int64_t OC_DATA_BLK = 16;
        static const int64_t MAX_S_REGS = 6;
        static const int64_t MAX_OC_REGS = 2;
        static const int64_t S_REG_ELTS = 1;
        static const int64_t OC_REG_ELTS = 8;
        static const int64_t OC_DATA_BLK_REGS = 2;
        static const int64_t MAX_OC_DATA_BLKS = MAX_OC_REGS / OC_DATA_BLK_REGS;
        static const int64_t MAX_S_BLK = MAX_S_REGS * S_REG_ELTS;
        static const int64_t MAX_OC_BLK = MAX_OC_DATA_BLKS * OC_DATA_BLK;
        static const int64_t NT_STORE_OPT = 2;
    };

    typedef int64_t flag_t;
    struct flag {
        static const flag_t LOAD_BIAS = (1 << 1);
        static const flag_t ADD_BIAS = (1 << 2);
        static const flag_t RELU = (1 << 11);
        static const flag_t RELU6 = (1 << 12);
    };

    conv2d_n16cx_gemm_direct_kernel_fp32_fma(int64_t *param) : param_(param) { }
    void set_param(int64_t *param) { this->param_ = param; }
    int64_t *param() { return param_; }

    void execute(const int64_t nt_store, const int64_t oc_reg, const int64_t s_reg) {
        table_[nt_store][oc_reg - 1][s_reg - 1](param_);
    }

private:
    int64_t *param_;
    static const func_t table_[config::NT_STORE_OPT][config::MAX_OC_REGS][config::MAX_S_REGS];
};

}}}; // namespace ppl::kernel::x86

#endif