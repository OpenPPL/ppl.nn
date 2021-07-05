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

#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_V2_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_V2_H_

#include <vector>

#include "ppl/kernel/x86/common/gemm_v2_common.h"

namespace ppl { namespace kernel { namespace x86 {

struct gemm_v2_algo_select_strategy {
    typedef enum {
        static_select  = 0,
        dynamic_select = 1, // TODO: need to be implemented
    } algo_type_select_strategy_t;
    typedef enum {
        use_default_param    = 0,
        static_select_param  = 1, // TODO: need to be implemented
        dynamic_select_param = 2, // TODO: need to be implemented
    } internal_param_select_strategy_t;

    algo_type_select_strategy_t algo_type_select_strategy           = static_select;
    internal_param_select_strategy_t internal_param_select_strategy = use_default_param;
};

class gemm_v2_fp32_algo_type {
public:
    enum {
        undef                               = 0,
        mnk_kernel_nm_atbn_fp32_avx512      = 1,
        mnk_sub_kmn_kernel_nm_atbn_fp32_fma = 2,
        mnk_sub_kmn_kernel_nm_atbn_fp32_sse = 3,
    };
};
typedef uint32_t gemm_v2_fp32_algo_type_t;

struct gemm_v2_algo_info_fp32 { // used for compiler selection & serialization
    gemm_v2_fp32_algo_type_t algo_type = gemm_v2_fp32_algo_type::undef;
    std::vector<uint8_t> internal_param;
};

class gemm_v2_executor_fp32 {
public:
    gemm_v2_executor_fp32() {}
    virtual ~gemm_v2_executor_fp32() {}

    void set_param(const gemm_v2_param_fp32& param)
    {
        param_ = param;
    }

    const gemm_v2_param_fp32& get_param(void)
    {
        return param_;
    }

    gemm_v2_param_fp32& get_param_mutable(void)
    {
        return param_;
    }

    virtual void set_internal_param(const std::vector<uint8_t>& internal_param) = 0;
    virtual const void* get_internal_param_ptr(void)                            = 0;
    virtual uint64_t get_internal_param_bytes(void)                             = 0;

    void set_temp_buffer(void* temp_buffer)
    {
        temp_buffer_ = temp_buffer;
    }

    virtual uint64_t get_buffer_bytes(void) const = 0;
    virtual ppl::common::RetCode execute(void)    = 0;
    virtual ppl::common::RetCode optimize(void)   = 0;

protected:
    gemm_v2_param_fp32 param_;
    void* temp_buffer_;
};

gemm_v2_executor_fp32* create_gemm_v2_executor_fp32(const gemm_v2_param_fp32& param);
gemm_v2_executor_fp32* create_gemm_v2_executor_fp32(const gemm_v2_param_fp32& param, const gemm_v2_algo_info_fp32& algo_info);
gemm_v2_executor_fp32* create_gemm_v2_executor_fp32(const gemm_v2_param_fp32& param, const gemm_v2_algo_info_fp32& algo_info, const gemm_v2_algo_select_strategy& algo_select_type);

}}} // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_GEMM_V2_H_