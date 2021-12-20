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

#include "ppl/kernel/x86/fp32/pd_conv2d.h"

#include "ppl/kernel/x86/fp32/pd_conv2d/fma/pd_conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/pd_conv2d/fma/pd_conv2d_n16cx_direct_ndarray_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/fma/conv2d_n16cx_direct_ndarray_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/fma/conv2d_n16cx_depthwise_fp32_fma.h"

#ifdef PPL_USE_X86_AVX512
#include "ppl/kernel/x86/fp32/pd_conv2d/avx512/pd_conv2d_n16cx_gemm_direct_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/avx512/conv2d_n16cx_gemm_direct_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/avx512/conv2d_n16cx_depthwise_fp32_avx512.h"
#endif

namespace ppl { namespace kernel { namespace x86 {

pd_conv2d_fp32_algo_info pd_conv2d_algo_selector::select_algo(
    const conv2d_fp32_algo_info &algo,
    const conv2d_fp32_algo_info &post_algo,
    const conv2d_fp32_param &param,
    const conv2d_fp32_param &post_param)
{
    if (true // gemm_direct algo
        && algo.algo_type == ppl::kernel::x86::conv2d_fp32_algo::GEMM_DIRECT
        && algo.input_format == ppl::common::DATAFORMAT_N16CX
        && algo.output_format == ppl::common::DATAFORMAT_N16CX
        && post_algo.algo_type == ppl::kernel::x86::conv2d_fp32_algo::DEPTHWISE
        && post_algo.input_format == ppl::common::DATAFORMAT_N16CX
        && post_algo.output_format == ppl::common::DATAFORMAT_N16CX)
    {
        if (algo.isa == ppl::common::ISA_X86_FMA && post_algo.isa == ppl::common::ISA_X86_FMA) {
            if (true // gemm_direct fma support param
                && !(param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM)
                && param.sparse_level() == 1.0f
                && param.group == 1
                && !(post_param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM)
                && post_param.dilation_h == 1
                && post_param.dilation_w == 1
                && param.num_output == post_param.channels) {
                return {
                    pd_conv2d_fp32_algo::GEMM_DIRECT,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

#ifdef PPL_USE_X86_AVX512
        if (algo.isa == ppl::common::ISA_X86_AVX512 && post_algo.isa == ppl::common::ISA_X86_AVX512) {
            if (true // gemm_direct fma support param
                && !(param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM)
                && param.sparse_level() == 1.0f
                && param.group == 1
                && !(post_param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM)
                && post_param.dilation_h == 1
                && post_param.dilation_w == 1
                && param.num_output == post_param.channels) {
                return {
                    pd_conv2d_fp32_algo::GEMM_DIRECT,
                    ppl::common::ISA_X86_AVX512,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }
#endif
    }

    if (true // direct_ndarray algo
        && algo.algo_type == ppl::kernel::x86::conv2d_fp32_algo::DIRECT
        && algo.input_format == ppl::common::DATAFORMAT_NDARRAY
        && algo.output_format == ppl::common::DATAFORMAT_N16CX
        && post_algo.algo_type == ppl::kernel::x86::conv2d_fp32_algo::DEPTHWISE
        && post_algo.input_format == ppl::common::DATAFORMAT_N16CX
        && post_algo.output_format == ppl::common::DATAFORMAT_N16CX)
    {
        if (algo.isa == ppl::common::ISA_X86_FMA && post_algo.isa == ppl::common::ISA_X86_FMA) {
            if (true // gemm_direct fma support param
                && !(param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM)
                && !(post_param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM)
                && post_param.dilation_h == 1
                && post_param.dilation_w == 1
                && param.num_output == post_param.channels) {
                return {
                    pd_conv2d_fp32_algo::DIRECT,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_NDARRAY,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }
    }

    return {
        pd_conv2d_fp32_algo::UNKNOWN,
        ppl::common::ISA_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN};
}

pd_conv2d_fp32_manager *pd_conv2d_algo_selector::gen_algo(
    const conv2d_fp32_param &param,
    const conv2d_fp32_param &depthwise_param,
    const pd_conv2d_fp32_algo_info &algo_info,
    ppl::common::Allocator *allocator)
{
    if (algo_info.algo_type == pd_conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new pd_conv2d_n16cx_gemm_direct_fp32_fma_manager(
            new conv2d_n16cx_gemm_direct_fp32_fma_manager(param, allocator),
            new conv2d_n16cx_depthwise_fp32_fma_manager(depthwise_param, allocator));
    }
    if (algo_info.algo_type == pd_conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new pd_conv2d_n16cx_direct_ndarray_fp32_fma_manager(
            new conv2d_n16cx_direct_ndarray_fp32_fma_manager(param, allocator),
            new conv2d_n16cx_depthwise_fp32_fma_manager(depthwise_param, allocator));
    }

#ifdef PPL_USE_X86_AVX512
    if (algo_info.algo_type == pd_conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new pd_conv2d_n16cx_gemm_direct_fp32_avx512_manager(
            new conv2d_n16cx_gemm_direct_fp32_avx512_manager(param, allocator),
            new conv2d_n16cx_depthwise_fp32_avx512_manager(depthwise_param, allocator));
    }
#endif

    return nullptr;
}

pd_conv2d_fp32_manager *pd_conv2d_algo_selector::gen_algo(
    const pd_conv2d_fp32_algo_info &algo_info,
    conv2d_fp32_manager *mgr,
    conv2d_fp32_manager *depthwise_mgr)
{
    if (algo_info.algo_type == pd_conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new pd_conv2d_n16cx_gemm_direct_fp32_fma_manager(mgr, depthwise_mgr);
    }
    if (algo_info.algo_type == pd_conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new pd_conv2d_n16cx_direct_ndarray_fp32_fma_manager(mgr, depthwise_mgr);
    }

#ifdef PPL_USE_X86_AVX512
    if (algo_info.algo_type == pd_conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new pd_conv2d_n16cx_gemm_direct_fp32_fma_manager(mgr, depthwise_mgr);
    }
#endif

    return nullptr;
}

}}};
