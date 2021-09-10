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
        static const int64_t a_ptr_idx = 0;
        static const int64_t packed_b_ptr_idx = 1;
        static const int64_t c_ptr_idx = 2;
        static const int64_t v_ptr_idx = 3;
        static const int64_t h_ptr_idx = 4;
        static const int64_t m_idx = 5;
        static const int64_t k_idx = 6;
        static const int64_t lda_idx = 7;
        static const int64_t ldpacked_b_idx = 8;
        static const int64_t ldc_idx = 9;
        static const int64_t ldh_idx = 10;
        static const int64_t flags_idx = 11;
        static const int64_t length = 12;
    };

    struct config {
        static const int64_t max_m_regs = 1;
        static const int64_t max_n_regs = 12;
        static const int64_t m_reg_elts = 1;
        static const int64_t n_reg_elts = 4;
        static const int64_t n_regb_regs = 2;
        static const int64_t max_n_regbs = max_n_regs / n_regb_regs;
        static const int64_t n_regb_elts = n_reg_elts * n_regb_regs;
        static const int64_t max_m_blk = max_m_regs;
        static const int64_t max_n_blk = max_n_regs * n_reg_elts;
        static const int64_t unroll_k = 8;
    };

    typedef int64_t flag_t;
    struct flag {
        static const flag_t load_h = (1 << 1);
        static const flag_t add_v = (1 << 2);
        static const flag_t relu = (1 << 11);
        static const flag_t relu6 = (1 << 12);
    };

    conv_gemm_kernel_fp32_sse(int64_t *param) : param_(param) { }
    void set_param(int64_t *param) { this->param_ = param; }
    int64_t *param() { return param_; }

    void execute(const int64_t n_regb) {
        table_[n_regb - 1](param_);
    }

private:
    int64_t *param_;
    static const func_t table_[config::max_n_regbs];
};

}}}; // namespace ppl::kernel::x86

#endif
