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

#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_sse_GEMM_KERNEL_FP32_sse_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_sse_GEMM_KERNEL_FP32_sse_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

class gemm_kernel_fp32_sse {
public:
    typedef void (*func_t)(int64_t*);

    struct param_def {
        static const int64_t A_PTR_IDX = 0;
        static const int64_t B_PTR_IDX = 1;
        static const int64_t C_PTR_IDX = 2;
        static const int64_t BIAS_PTR_IDX = 3;
        static const int64_t SUM_PTR_IDX = 4;
        static const int64_t M_IDX = 5;
        static const int64_t K_IDX = 6;
        static const int64_t LDC_IDX = 7;
        static const int64_t LDSUM_IDX = 8;
        static const int64_t ALPHA_IDX = 9;
        static const int64_t BETA_IDX = 10;
        static const int64_t BETA_BIAS_IDX = 11;
        static const int64_t BETA_SUM_IDX = 12;
        static const int64_t FLAGS_IDX = 13;
        static const int64_t PRF_C_LDK_IDX = 14;
        static const int64_t NEXT_B_PTR_IDX = 15;
        static const int64_t MASK_IDX = 16;
        static const int64_t MASKED_C_IDX = 17;
        static const int64_t MASKED_C_LENGTH = 4;
        static const int64_t MASKED_SUM_IDX = 21;
        static const int64_t MASKED_SUM_LENGTH = 4;
        static const int64_t MASKED_BIAS_IDX = 25;
        static const int64_t MASKED_BIAS_LENGTH = 4;
        static const int64_t SIX_IDX = 29;
        static const int64_t SIX_LENGTH = 2;
        static const int64_t LENGTH = 32;
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
        static const int64_t NEED_MASK_OPT = 2;
        static const int64_t PRF_C_LDK_MEM = 64;
        static const int64_t PRF_C_LDK_L3  = 32;
    };

    typedef int64_t flag_t;
    struct flag {
        static const flag_t LOAD_C = (1 << 1);
        static const flag_t WITH_SUM = (1 << 2);
        static const flag_t ROW_BIAS = (1 << 3);
        static const flag_t COL_BIAS = (1 << 4);
        static const flag_t SCA_BIAS = (1 << 5);
        static const flag_t RELU = (1 << 11);
        static const flag_t RELU6 = (1 << 12);
    };

    gemm_kernel_fp32_sse(int64_t *param) { set_param(param); }
    inline void set_param(int64_t *param) { 
        this->param_ = param;
        auto six = (float*)&param[param_def::SIX_IDX];
        six[0] = 6.0f; six[1] = 6.0f;
        six[2] = 6.0f; six[3] = 6.0f;
    }

    inline int64_t *param() { return param_; }

    inline void execute(const int64_t need_mask, const int64_t m_reg, const int64_t n_regb) {
        table_[need_mask][n_regb - 1][m_reg - 1](param_);
    }

private:
    int64_t *param_;
    static const func_t table_[config::NEED_MASK_OPT][config::MAX_N_REGBS][config::MAX_M_REGS];
};

}}}; // namespace ppl::kernel::x86

#endif
