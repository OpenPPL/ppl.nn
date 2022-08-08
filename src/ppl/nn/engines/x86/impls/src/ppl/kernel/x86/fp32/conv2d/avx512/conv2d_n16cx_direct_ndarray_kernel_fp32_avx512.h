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

namespace ppl { namespace kernel { namespace x86 {

class conv2d_n16cx_direct_ndarray_kernel_fp32_avx512 {
public:
    typedef void (*func_t)(int64_t*);

    struct param_def {
        static const int64_t SRC_PTR_IDX = 0;
        static const int64_t SUM_SRC_PTR_IDX = 1;
        static const int64_t DST_PTR_IDX = 2;
        static const int64_t FLT_PTR_IDX = 3;
        static const int64_t BIAS_PTR_IDX = 4;
        static const int64_t DST_WIDTH_IDX = 5;
        static const int64_t CHANNELS_IDX = 6;
        static const int64_t SRC_H_STRIDE_IDX = 7;
        static const int64_t SRC_C_STRIDE_IDX = 8;
        static const int64_t FLT_C_STRIDE_IDX = 9;
        static const int64_t SUM_SRC_OCB_STRIDE_IDX = 10;
        static const int64_t DST_OCB_STRIDE_IDX = 11;
        static const int64_t FLT_OCB_STRIDE_IDX = 12;
        static const int64_t KH_START_IDX = 13;
        static const int64_t KH_END_IDX = 14;
        static const int64_t KW_START_IDX = 15;
        static const int64_t KW_END_IDX = 16;
        static const int64_t KH_IDX = 17;
        static const int64_t KW_IDX = 18;
        static const int64_t SW_IDX = 19;
        static const int64_t FLAGS_IDX = 20;
        static const int64_t LENGTH = 21;
    };

    struct config {
        static const int64_t OC_DATA_BLK = 16;
        static const int64_t MAX_W_REGS = 14;
        static const int64_t MAX_OC_REGS = 2;
        static const int64_t W_REG_ELTS = 1;
        static const int64_t OC_REG_ELTS = 16;
        static const int64_t OC_DATA_BLK_REGS = 1;
        static const int64_t MAX_OC_DATA_BLKS = MAX_OC_REGS / OC_DATA_BLK_REGS;
        static const int64_t MAX_W_BLK = MAX_W_REGS * W_REG_ELTS;
        static const int64_t MAX_OC_BLK = MAX_OC_DATA_BLKS * OC_DATA_BLK;
        static const int64_t NT_STORE_OPT = 2;
    };

    typedef int64_t flag_t;
    struct flag {
        static const flag_t SUM = (1 << 10);
        static const flag_t RELU = (1 << 11);
        static const flag_t RELU6 = (1 << 12);
    };

    conv2d_n16cx_direct_ndarray_kernel_fp32_avx512(int64_t *param) : param_(param) { }
    inline void set_param(int64_t *param) { this->param_ = param; }
    inline int64_t *param() { return param_; }

    inline void execute(const int64_t nt_store, const int64_t oc_reg, const int64_t w_reg) {
        table_[nt_store][oc_reg - 1][w_reg - 1](param_);
    }

    inline void execute_border(const int64_t nt_store, const int64_t oc_reg) {
        border_table_[nt_store][oc_reg - 1](param_);
    }

private:
    int64_t *param_;
    static const func_t table_[config::NT_STORE_OPT][config::MAX_OC_REGS][config::MAX_W_REGS];
    static const func_t border_table_[config::NT_STORE_OPT][config::MAX_OC_REGS];
};

}}}; // namespace ppl::kernel::x86

#endif
