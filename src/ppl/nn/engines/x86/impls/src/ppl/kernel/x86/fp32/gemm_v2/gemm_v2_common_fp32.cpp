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

#include "ppl/kernel/x86/fp32/gemm_v2.h"
#ifdef PPLNN_USE_X86_AVX512
#include "ppl/kernel/x86/fp32/gemm_v2/avx512/gemm_v2_mnk_kernel_nm_atbn_executor_fp32_avx512.h"
#endif
#include "ppl/kernel/x86/fp32/gemm_v2/fma/gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_fma.h"
#include "ppl/kernel/x86/fp32/gemm_v2/sse/gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

gemm_v2_executor_fp32* create_gemm_v2_executor_fp32(
    const gemm_v2_param_fp32& param,
    const gemm_v2_algo_info_fp32& algo_info,
    const gemm_v2_algo_select_strategy& algo_select_strategy)
{
    gemm_v2_executor_fp32* executor = nullptr;

    // algo has been selected
    if (algo_info.algo_type != gemm_v2_fp32_algo_type::undef) {
        if (false) {
        }
#ifdef PPLNN_USE_X86_AVX512
        else if (algo_info.algo_type == gemm_v2_fp32_algo_type::mnk_kernel_nm_atbn_fp32_avx512) {
            executor = new gemm_v2_mnk_kernel_nm_atbn_executor_fp32_avx512;
        }
#endif
        else if (algo_info.algo_type == gemm_v2_fp32_algo_type::mnk_sub_kmn_kernel_nm_atbn_fp32_fma) {
            executor = new gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_fma;
        } else if (algo_info.algo_type == gemm_v2_fp32_algo_type::mnk_sub_kmn_kernel_nm_atbn_fp32_sse) {
            executor = new gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse;
        }

        if (executor) {
            executor->set_internal_param(algo_info.internal_param); // copy compiler stage's internal param to executor
        }
    } else { // algo not selected, select according to param & algo_select_type
        if (algo_select_strategy.algo_type_select_strategy == gemm_v2_algo_select_strategy::static_select && // only support static_select & use_default_param now
            algo_select_strategy.internal_param_select_strategy == gemm_v2_algo_select_strategy::use_default_param) {
            if (false) {
            }
#ifdef PPLNN_USE_X86_AVX512
            else if (param.isa_flag & common::ISA_X86_AVX512) {
                executor = new gemm_v2_mnk_kernel_nm_atbn_executor_fp32_avx512;
            }
#endif
            else if (param.isa_flag & common::ISA_X86_FMA) {
                executor = new gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_fma;
            } else if (param.isa_flag & common::ISA_X86_SSE) {
                executor = new gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse;
            }
        }
        // TODO: finished other algo_select_type
    }

    if (executor) {
        executor->set_param(param);
    }
    return executor;
}

gemm_v2_executor_fp32* create_gemm_v2_executor_fp32(const gemm_v2_param_fp32& param, const gemm_v2_algo_info_fp32& algo_info)
{
    gemm_v2_algo_select_strategy default_strategy;
    return create_gemm_v2_executor_fp32(param, algo_info, default_strategy);
}

gemm_v2_executor_fp32* create_gemm_v2_executor_fp32(const gemm_v2_param_fp32& param)
{
    gemm_v2_algo_info_fp32 empty_algo_info;
    gemm_v2_algo_select_strategy default_strategy;
    return create_gemm_v2_executor_fp32(param, empty_algo_info, default_strategy);
}

}}} // namespace ppl::kernel::x86