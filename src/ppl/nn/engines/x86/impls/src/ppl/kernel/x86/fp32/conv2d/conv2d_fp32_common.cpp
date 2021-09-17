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

#include <new>

#include "ppl/kernel/x86/fp32/conv2d.h"

#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/winograd/fma/conv2d_n16cx_winograd_b4f3_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/fma/conv2d_n16cx_depthwise_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/fma/conv2d_im2col_gemm_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/fma/conv2d_n16cx_direct_ndarray_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_v2_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/direct/fma/conv2d_n16cx_direct_v2_fp32_fma.h"

#ifdef PPL_USE_X86_AVX512
#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/avx512/conv2d_n16cx_gemm_direct_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/avx512/conv2d_n16cx_depthwise_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_b4f3_fp32_avx512.h"
#endif

#include "ppl/kernel/x86/fp32/conv2d/direct/sse/conv2d_n8cx_direct_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/sse/conv2d_n8cx_gemm_direct_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_n8cx_depthwise_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/sse/conv2d_n8cx_direct_ndarray_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/sse/conv2d_im2col_gemm_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_depthwise_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

conv2d_fp32_algo_info conv2d_algo_selector::select_algo(const ppl::common::dataformat_t src_format, const conv2d_fp32_param &param, const ppl::common::isa_t isa_flags)
{
    static conv2d_fp32_algo_info unknown_info = {
        conv2d_fp32_algo::unknown,
        ppl::common::ISA_undef,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN};

    static conv2d_fp32_algo_info fma_fallback_info = {
        conv2d_fp32_algo::im2col_gemm,
        ppl::common::ISA_X86_FMA,
        ppl::common::DATAFORMAT_NDARRAY,
        ppl::common::DATAFORMAT_NDARRAY};

    static conv2d_fp32_algo_info sse_fallback_info = {
        conv2d_fp32_algo::im2col_gemm,
        ppl::common::ISA_X86_SSE,
        ppl::common::DATAFORMAT_NDARRAY,
        ppl::common::DATAFORMAT_NDARRAY};


#ifdef PPL_USE_X86_AVX512
    if (isa_flags & ppl::common::ISA_X86_AVX512) {
        if (src_format == ppl::common::DATAFORMAT_NDARRAY) {
            auto direct_ndarray_mgr = new conv2d_n16cx_direct_ndarray_fp32_avx512_manager(param, nullptr);
            bool supported          = direct_ndarray_mgr->is_supported();
            delete direct_ndarray_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::direct,
                    ppl::common::ISA_X86_AVX512,
                    ppl::common::DATAFORMAT_NDARRAY,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (param.is_depthwise()) {
            auto dw_mgr    = new conv2d_n16cx_depthwise_fp32_avx512_manager(param, nullptr);
            bool supported = dw_mgr->is_supported();
            delete dw_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::depthwise,
                    ppl::common::ISA_X86_AVX512,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (param.is_pointwise()) {
            auto gd_mgr    = new conv2d_n16cx_gemm_direct_fp32_avx512_manager(param, nullptr);
            bool supported = gd_mgr->is_supported();
            delete gd_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::gemm_direct,
                    ppl::common::ISA_X86_AVX512,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (!param.is_depthwise() &&
            param.kernel_h == 3 && param.kernel_w == 3 &&
            param.stride_h == 1 && param.stride_w == 1 &&
            param.dilation_h == 1 && param.dilation_w == 1) {
            auto wg_mgr    = new conv2d_n16cx_winograd_b4f3_fp32_avx512_manager(param, nullptr);
            bool supported = wg_mgr->is_supported();
            delete wg_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::winograd_b4f3,
                    ppl::common::ISA_X86_AVX512,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        {
            auto direct_mgr = new conv2d_n16cx_direct_fp32_avx512_manager(param, nullptr);
            bool supported  = direct_mgr->is_supported();
            delete direct_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::direct,
                    ppl::common::ISA_X86_AVX512,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }
    }
#endif

    if (isa_flags & ppl::common::ISA_X86_FMA) {
        if (src_format == ppl::common::DATAFORMAT_NDARRAY) {
            auto direct_ndarray_mgr = new conv2d_n16cx_direct_ndarray_fp32_fma_manager(param, nullptr);
            bool supported          = direct_ndarray_mgr->is_supported();
            delete direct_ndarray_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::direct,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_NDARRAY,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (param.is_depthwise()) {
            auto dw_mgr    = new conv2d_n16cx_depthwise_fp32_fma_manager(param, nullptr);
            bool supported = dw_mgr->is_supported();
            delete dw_mgr;
            if (!supported) {
                return fma_fallback_info;
            } else {
                return {
                    conv2d_fp32_algo::depthwise,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (param.is_pointwise()) {
            auto gd_mgr    = new conv2d_n16cx_gemm_direct_v2_fp32_fma_manager(param, nullptr);
            bool supported = gd_mgr->is_supported();
            delete gd_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::gemm_direct_v2,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (!param.is_depthwise() &&
            param.kernel_h == 3 && param.kernel_w == 3 &&
            param.stride_h == 1 && param.stride_w == 1 &&
            param.dilation_h == 1 && param.dilation_w == 1) {
            auto wg_mgr    = new conv2d_n16cx_winograd_b4f3_fp32_fma_manager(param, nullptr);
            bool supported = wg_mgr->is_supported();
            delete wg_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::winograd_b4f3,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        {
            auto direct_mgr = new conv2d_n16cx_direct_v2_fp32_fma_manager(param, nullptr);
            bool supported  = direct_mgr->is_supported();
            delete direct_mgr;
            if (!supported) {
                return fma_fallback_info;
            } else {
                return {
                    conv2d_fp32_algo::direct_v2,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }
    }

    if (isa_flags & ppl::common::ISA_X86_SSE) {
        if (param.is_depthwise()) {
            auto dw_mgr    = new conv2d_depthwise_fp32_sse_manager(param, nullptr);
            bool supported = dw_mgr->is_supported();
            delete dw_mgr;
            if (!supported) {
                return sse_fallback_info;
            } else {
                return {
                    conv2d_fp32_algo::depthwise,
                    ppl::common::ISA_X86_SSE,
                    ppl::common::DATAFORMAT_NDARRAY,
                    ppl::common::DATAFORMAT_NDARRAY};
            }
        }

        return sse_fallback_info;
    }

    return unknown_info;
}

conv2d_fp32_manager *conv2d_algo_selector::gen_algo(const conv2d_fp32_param &param, const conv2d_fp32_algo_info &algo_info, ppl::common::Allocator *allocator)
{
    conv2d_fp32_manager *conv_mgr = nullptr;
    if (algo_info.algo_type == conv2d_fp32_algo::gemm_direct &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_gemm_direct_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::gemm_direct_v2 &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_gemm_direct_v2_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::depthwise &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_depthwise_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::winograd_b4f3 &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_winograd_b4f3_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::direct_v2 &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_direct_v2_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::direct &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_direct_ndarray_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::im2col_gemm &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        conv_mgr = new conv2d_im2col_gemm_fp32_fma_manager(param, allocator);
    }
#ifdef PPL_USE_X86_AVX512
    if (algo_info.algo_type == conv2d_fp32_algo::winograd_b4f3 &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_winograd_b4f3_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::direct &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_direct_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::direct &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_direct_ndarray_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::gemm_direct &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_gemm_direct_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::depthwise &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        conv_mgr = new conv2d_n16cx_depthwise_fp32_avx512_manager(param, allocator);
    }
#endif
    if (algo_info.algo_type == conv2d_fp32_algo::direct &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_direct_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::gemm_direct &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_gemm_direct_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::depthwise &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_depthwise_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::direct &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_direct_ndarray_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::im2col_gemm &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        conv_mgr = new conv2d_im2col_gemm_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::depthwise &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        conv_mgr = new conv2d_depthwise_fp32_sse_manager(param, allocator);
    }

    return conv_mgr;
}

}}}; // namespace ppl::kernel::x86
