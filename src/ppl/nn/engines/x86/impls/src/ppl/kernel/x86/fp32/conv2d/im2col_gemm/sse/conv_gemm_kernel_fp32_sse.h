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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_SSE_CONV_GEMM_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_SSE_CONV_GEMM_KERNEL_FP32_SSE_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

// packed_b layout is Nk8n
class conv_gemm_kernel_fp32_sse {
public:
    typedef void (*func_t)(int64_t*);

    struct param_def {
        static const int64_t A_PTR_IDX = 0;
        static const int64_t PACKED_B_PTR_IDX = 1;
        static const int64_t C_PTR_IDX = 2;
        static const int64_t V_PTR_IDX = 3;
        static const int64_t H_PTR_IDX = 4;
        static const int64_t M_IDX = 5;
        static const int64_t K_IDX = 6;
        static const int64_t LDA_IDX = 7;
        static const int64_t LDPACKED_B_IDX = 8;
        static const int64_t LDC_IDX = 9;
        static const int64_t LDH_IDX = 10;
        static const int64_t FLAGS_IDX = 11;
        static const int64_t LENGTH = 12;
    };

    struct config {
        static const int64_t MAX_M_REGS = 1;
        static const int64_t MAX_N_REGS = 12;
        static const int64_t M_REG_ELTS = 1;
        static const int64_t N_REG_ELTS = 4;
        static const int64_t N_REGB_REGS = 2;
        static const int64_t MAX_N_REGBS = MAX_N_REGS / N_REGB_REGS;
        static const int64_t N_REGB_ELTS = N_REG_ELTS * N_REGB_REGS;
        static const int64_t MAX_M_BLK = MAX_M_REGS * M_REG_ELTS;
        static const int64_t MAX_N_BLK = MAX_N_REGS * N_REG_ELTS;
        static const int64_t UNROLL_K = 8;
    };

    typedef int64_t flag_t;
    struct flag {
        static const flag_t LOAD_H = (1 << 1);
        static const flag_t ADD_V = (1 << 2);
        static const flag_t RELU = (1 << 11);
        static const flag_t RELU6 = (1 << 12);
    };

    conv_gemm_kernel_fp32_sse(int64_t *param) : param_(param) { }
    void set_param(int64_t *param) { this->param_ = param; }
    int64_t *param() { return param_; }

    void execute(const int64_t n_regb) {
        table_[n_regb - 1](param_);
    }

private:
    int64_t *param_;
    static const func_t table_[config::MAX_N_REGBS];
};

}}}; // namespace ppl::kernel::x86

#endif
