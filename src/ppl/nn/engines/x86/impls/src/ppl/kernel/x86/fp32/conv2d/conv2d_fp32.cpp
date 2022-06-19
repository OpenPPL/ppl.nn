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
#include "ppl/kernel/x86/fp32/conv2d/direct/fma/conv2d_n16cx_direct_fp32_fma.h"

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
#include "ppl/kernel/x86/fp32/conv2d/winograd/sse/conv2d_winograd_b6f3_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode conv2d_fp32_ref(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *sum_src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float *sum_src,
    const float *filter,
    const float *bias,
    const conv2d_fp32_param &param,
    float *dst)
{
    const int64_t batch      = src_shape->GetDim(0);
    const int64_t src_c      = src_shape->GetDim(1);
    const int64_t src_h      = src_shape->GetDim(2);
    const int64_t src_w      = src_shape->GetDim(3);
    const int64_t dst_c      = dst_shape->GetDim(1);
    const int64_t dst_h      = dst_shape->GetDim(2);
    const int64_t dst_w      = dst_shape->GetDim(3);
    const int64_t ic_per_gp  = param.channels / param.group;
    const int64_t oc_per_gp  = param.num_output / param.group;
    const int64_t kernel_h   = param.kernel_h;
    const int64_t kernel_w   = param.kernel_w;
    const int64_t stride_h   = param.stride_h;
    const int64_t stride_w   = param.stride_w;
    const int64_t pad_h      = param.pad_h;
    const int64_t pad_w      = param.pad_w;
    const int64_t dilation_h = param.dilation_h;
    const int64_t dilation_w = param.dilation_w;

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < param.group; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t oc = 0; oc < oc_per_gp; ++oc) {
                for (int64_t oh = 0; oh < dst_h; ++oh) {
                    const float *filter_d = filter + g * oc_per_gp * ic_per_gp * kernel_h * kernel_w;
                    const float *input_d  = src + (b * src_c + g * ic_per_gp) * src_h * src_w;
                    float *output_d       = dst + (b * dst_c + g * oc_per_gp) * dst_h * dst_w;
                    int64_t output_idx    = oc * dst_h * dst_w + oh * dst_w;
                    for (int64_t ow = 0; ow < dst_w; ++ow) {
                        const int64_t ih_start = -pad_h + oh * stride_h;
                        const int64_t iw_start = -pad_w + ow * stride_w;
                        int64_t flt_idx        = oc * ic_per_gp * kernel_h * kernel_w;
                        float sum_val          = 0.0f;
                        for (int64_t ic = 0; ic < ic_per_gp; ++ic) {
                            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                                const int64_t ih   = ih_start + dilation_h * kh;
                                const bool valid_h = (ih >= 0 && ih < src_h);
                                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                    const int64_t iw   = iw_start + dilation_w * kw;
                                    const bool valid_w = (iw >= 0 && iw < src_w);
                                    if (valid_h && valid_w) {
                                        const int64_t input_idx = ic * src_h * src_w + ih * src_w + iw;
                                        sum_val += filter_d[flt_idx] * input_d[input_idx];
                                    }
                                    ++flt_idx;
                                }
                            }
                        }
                        if (bias != nullptr) {
                            sum_val += bias[g * oc_per_gp + oc];
                        }
                        if (param.fuse_flag & conv_fuse_flag::SUM) {
                            const float *sum_d = sum_src + (b * sum_src_shape->GetDim(1) + g * oc_per_gp) * dst_h * dst_w;
                            sum_val += sum_d[output_idx];
                        }
                        if (param.fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
                            sum_val = max(sum_val, 0.0f);
                        }
                        if (param.fuse_flag & conv_fuse_flag::RELU6) {
                            sum_val = min(sum_val, 6.0f);
                        }
                        output_d[output_idx] = sum_val;
                        ++output_idx;
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

conv2d_fp32_algo_info conv2d_algo_selector::select_algo(const ppl::common::dataformat_t src_format, const conv2d_fp32_param &param, const ppl::common::isa_t isa_flags)
{
    static conv2d_fp32_algo_info unknown_info = {
        conv2d_fp32_algo::UNKNOWN,
        ppl::common::ISA_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN};

    static conv2d_fp32_algo_info fma_fallback_info = {
        conv2d_fp32_algo::IM2COL_GEMM,
        ppl::common::ISA_X86_FMA,
        ppl::common::DATAFORMAT_NDARRAY,
        ppl::common::DATAFORMAT_NDARRAY};

    static conv2d_fp32_algo_info sse_fallback_info = {
        conv2d_fp32_algo::IM2COL_GEMM,
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
                    conv2d_fp32_algo::DIRECT,
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
                    conv2d_fp32_algo::DEPTHWISE,
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
                    conv2d_fp32_algo::GEMM_DIRECT,
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
                    conv2d_fp32_algo::WINOGRAD_B4F3,
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
                    conv2d_fp32_algo::DIRECT,
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
                    conv2d_fp32_algo::DIRECT,
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
                    conv2d_fp32_algo::DEPTHWISE,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        if (param.is_pointwise()) {
            auto gd_mgr    = new conv2d_n16cx_gemm_direct_fp32_fma_manager(param, nullptr);
            bool supported = gd_mgr->is_supported();
            delete gd_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::GEMM_DIRECT,
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
                    conv2d_fp32_algo::WINOGRAD_B4F3,
                    ppl::common::ISA_X86_FMA,
                    ppl::common::DATAFORMAT_N16CX,
                    ppl::common::DATAFORMAT_N16CX};
            }
        }

        {
            auto direct_mgr = new conv2d_n16cx_direct_fp32_fma_manager(param, nullptr);
            bool supported  = direct_mgr->is_supported();
            delete direct_mgr;
            if (!supported) {
                return fma_fallback_info;
            } else {
                return {
                    conv2d_fp32_algo::DIRECT,
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
                    conv2d_fp32_algo::DEPTHWISE,
                    ppl::common::ISA_X86_SSE,
                    ppl::common::DATAFORMAT_NDARRAY,
                    ppl::common::DATAFORMAT_NDARRAY};
            }
        }

        if (!param.is_depthwise() &&
            param.kernel_h == 3 && param.kernel_w == 3 &&
            param.stride_h == 1 && param.stride_w == 1 &&
            param.dilation_h == 1 && param.dilation_w == 1) {
            auto wg_mgr    = new conv2d_winograd_b6f3_fp32_sse_manager(param, nullptr);
            bool supported = wg_mgr->is_supported();
            delete wg_mgr;
            if (supported) {
                return {
                    conv2d_fp32_algo::WINOGRAD_B6F3,
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
    if (algo_info.algo_type == conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_gemm_direct_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DEPTHWISE &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_depthwise_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::WINOGRAD_B4F3 &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_winograd_b4f3_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_direct_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_direct_ndarray_fp32_fma_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::IM2COL_GEMM &&
        algo_info.isa == ppl::common::ISA_X86_FMA &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        return new conv2d_im2col_gemm_fp32_fma_manager(param, allocator);
    }
#ifdef PPL_USE_X86_AVX512
    if (algo_info.algo_type == conv2d_fp32_algo::WINOGRAD_B4F3 &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_winograd_b4f3_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_direct_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_direct_ndarray_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_gemm_direct_fp32_avx512_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DEPTHWISE &&
        algo_info.isa == ppl::common::ISA_X86_AVX512 &&
        algo_info.input_format == ppl::common::DATAFORMAT_N16CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N16CX) {
        return new conv2d_n16cx_depthwise_fp32_avx512_manager(param, allocator);
    }
#endif
    if (algo_info.algo_type == conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        return new conv2d_n8cx_direct_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::GEMM_DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        return new conv2d_n8cx_gemm_direct_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DEPTHWISE &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        return new conv2d_n8cx_depthwise_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DIRECT &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        return new conv2d_n8cx_direct_ndarray_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::IM2COL_GEMM &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        return new conv2d_im2col_gemm_fp32_sse_manager(param, allocator);
    }
    if (algo_info.algo_type == conv2d_fp32_algo::DEPTHWISE &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        return new conv2d_depthwise_fp32_sse_manager(param, allocator);
    }

    if (algo_info.algo_type == conv2d_fp32_algo::WINOGRAD_B6F3 &&
        algo_info.isa == ppl::common::ISA_X86_SSE &&
        algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
        algo_info.output_format == ppl::common::DATAFORMAT_NDARRAY) {
        return new conv2d_winograd_b6f3_fp32_sse_manager(param, allocator);
    }

    return nullptr;
}

}}}; // namespace ppl::kernel::x86
