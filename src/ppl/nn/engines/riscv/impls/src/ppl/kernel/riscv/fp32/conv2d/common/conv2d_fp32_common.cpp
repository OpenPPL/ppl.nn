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
#include <chrono>

#include "ppl/kernel/riscv/fp32/conv2d/naive/conv2d_ndarray_naive_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/tile_gemm/vec128/conv2d_ndarray_tile_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/tile_gemm/vec128/conv2d_n4cx_tile_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/gemm/conv2d_n4cx_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/direct_gemm/vec128/conv2d_n4cx_direct_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/wg/vec128/conv2d_n4cx_wg_b2f3_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/wg/vec128/conv2d_n4cx_wg_b4f3_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/wg/vec128/conv2d_n4cx_wg_b6f3_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/depthwise/vec128/conv2d_n4cx_dw_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d.h"
#include "ppl/common/log.h"
#include "ppl/common/types.h"

using namespace ppl::common;

namespace ppl { namespace kernel { namespace riscv {

conv2d_common_algo_info conv2d_fp32_algo_selector::select_best_algo(const void* filter, ppl::nn::TensorShape& src_shape, ppl::nn::TensorShape& dst_shape, const conv2d_common_param& param, Allocator* allocator, const ppl::nn::riscv::EngineOptions* engine_options)
{
    static conv2d_common_algo_info unknown_info =
        {conv2d_common_algo::unknown, DATAFORMAT_UNKNOWN, DATAFORMAT_UNKNOWN, DATATYPE_FLOAT32, DATATYPE_FLOAT32};

    static conv2d_common_algo_info ndarray_algo_info_lst[] = {
        {conv2d_common_algo::tile_gemm, DATAFORMAT_NDARRAY, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32}};

    static conv2d_common_algo_info n4cx_algo_info_lst[] = {
        {conv2d_common_algo::tile_gemm, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32},
        {conv2d_common_algo::gemm, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32},
        {conv2d_common_algo::winograd_b2f3, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32},
        {conv2d_common_algo::winograd_b4f3, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32},
        {conv2d_common_algo::winograd_b6f3, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32}};

    if (param.group == param.num_output && param.num_output == param.channels) {
        return {conv2d_common_algo::depthwise, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
    }

    std::vector<conv2d_common_algo_info> profiling_algo_info_vec;
    if (DATAFORMAT_NDARRAY == src_shape.GetDataFormat()) {
        for (auto algo_info : ndarray_algo_info_lst) {
            profiling_algo_info_vec.push_back(algo_info);
        }
    } else if (DATAFORMAT_N4CX == src_shape.GetDataFormat()) {
        for (auto algo_info : n4cx_algo_info_lst) {
            if (algo_info.algo_type == conv2d_common_algo::winograd_b2f3) {
                if ((ppl::nn::riscv::WG_ON != engine_options->winograd_level && ppl::nn::riscv::WG_ON_B2 != engine_options->winograd_level) ||
                    param.kernel_h != 3 || param.kernel_w != 3 || param.stride_h != 1 || param.stride_w != 1 ||
                    param.dilation_h != 1 || param.dilation_w != 1) {
                    continue;
                }
            } else if (algo_info.algo_type == conv2d_common_algo::winograd_b4f3) {
                if ((ppl::nn::riscv::WG_ON != engine_options->winograd_level && ppl::nn::riscv::WG_ON_B4 != engine_options->winograd_level) ||
                    param.kernel_h != 3 || param.kernel_w != 3 || param.stride_h != 1 || param.stride_w != 1 ||
                    param.dilation_h != 1 || param.dilation_w != 1) {
                    continue;
                }
            } else if (algo_info.algo_type == conv2d_common_algo::winograd_b6f3) {
                if ((ppl::nn::riscv::WG_ON != engine_options->winograd_level && ppl::nn::riscv::WG_ON_B6 != engine_options->winograd_level) ||
                    param.kernel_h != 3 || param.kernel_w != 3 || param.stride_h != 1 || param.stride_w != 1 ||
                    param.dilation_h != 1 || param.dilation_w != 1) {
                    continue;
                }
            } else if (algo_info.algo_type == conv2d_common_algo::gemm) {
                if (param.dilation_h != 1 || param.dilation_w != 1) {
                    continue;
                }
            }

            profiling_algo_info_vec.push_back(algo_info);
        }
    }

    const int32_t exe_count                = 1;
    double best_time                       = DBL_MAX;
    conv2d_common_algo_info best_algo_info = unknown_info;

    for (auto algo_info : profiling_algo_info_vec) {
        conv2d_offline_manager<float>* conv_manager = gen_algo(param, algo_info, allocator);
        auto ori_input_format                       = src_shape.GetDataFormat();
        auto ori_output_format                      = dst_shape.GetDataFormat();
        src_shape.SetDataFormat(algo_info.input_format);
        dst_shape.SetDataFormat(algo_info.output_format);
        std::vector<float> dst(dst_shape.CalcElementsIncludingPadding(), 0.f);
        std::vector<float> src(src_shape.CalcElementsIncludingPadding(), 0.f);

        if (conv_manager == nullptr) {
            return algo_info;
        }

        double profiling_time = conv_manager->profile_tunning_param(src.data(), (const float*)filter, dst.data(), src_shape, dst_shape, exe_count);
        src_shape.SetDataFormat(ori_input_format);
        dst_shape.SetDataFormat(ori_output_format);
        src.resize(0);
        dst.resize(0);

        if (profiling_time < best_time) {
            best_time      = profiling_time;
            best_algo_info = algo_info;
        }
        delete conv_manager;
    }

    LOG(DEBUG) << "select best fp32 conv algo " << best_algo_info.algo_type;
    if (best_algo_info.algo_type == conv2d_common_algo::unknown) {
        best_algo_info = select_algo(src_shape, param, engine_options);
    }
    return best_algo_info;
}

conv2d_common_algo_info conv2d_fp32_algo_selector::select_algo(const ppl::nn::TensorShape& input_shape,
                                                               const conv2d_common_param& param,
                                                               const ppl::nn::riscv::EngineOptions* engine_options)
{
    LOG(DEBUG) << "RISCV FP32 CONV select algo";

    static conv2d_common_algo_info unknown_info =
        {conv2d_common_algo::unknown, DATAFORMAT_UNKNOWN, DATAFORMAT_UNKNOWN, DATATYPE_FLOAT32, DATATYPE_FLOAT32};

    if (DATAFORMAT_NDARRAY == input_shape.GetDataFormat()) {
        if (param.group == 1) {
            return {conv2d_common_algo::tile_gemm, DATAFORMAT_NDARRAY, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
        }
    }

    if (DATAFORMAT_N4CX == input_shape.GetDataFormat()) {
        if (param.group == 1 && param.kernel_h == 1 && param.kernel_w == 1 &&
            param.pad_h == 0 && param.pad_w == 0 &&
            param.stride_h == 1 && param.stride_w == 1 &&
            param.dilation_h == 1 && param.dilation_w == 1) {
            return {conv2d_common_algo::gemm, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
        }
        if (param.group == param.num_output && param.num_output == param.channels &&
            param.dilation_h == 1 && param.dilation_w == 1) {
            return {conv2d_common_algo::depthwise, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
        }

        if (param.kernel_h == 3 && param.kernel_w == 3 &&
            param.stride_h == 1 && param.stride_w == 1 &&
            param.dilation_h == 1 && param.dilation_w == 1) {
            if (ppl::nn::riscv::WG_OFF == engine_options->winograd_level) {
                return {conv2d_common_algo::tile_gemm, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
            } else if (ppl::nn::riscv::WG_ON_B2 == engine_options->winograd_level) {
                return {conv2d_common_algo::winograd_b2f3, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
            } else if (ppl::nn::riscv::WG_ON == engine_options->winograd_level || ppl::nn::riscv::WG_ON_B4 == engine_options->winograd_level) {
                return {conv2d_common_algo::winograd_b4f3, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
            } else if (ppl::nn::riscv::WG_ON_B6 == engine_options->winograd_level) {
                return {conv2d_common_algo::winograd_b6f3, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
            }
        }

        return {conv2d_common_algo::tile_gemm, DATAFORMAT_N4CX, DATAFORMAT_N4CX, DATATYPE_FLOAT32, DATATYPE_FLOAT32};
    }

    return unknown_info;
}

conv2d_offline_manager<float>* conv2d_fp32_algo_selector::gen_algo(const conv2d_common_param& param,
                                                                   const conv2d_common_algo_info& algo_info,
                                                                   Allocator* allocator)
{
    LOG(DEBUG) << "RISCV FP32 CONV gen algo";
    conv2d_offline_manager<float>* conv_mgr = nullptr;

    if (conv2d_common_algo::naive == algo_info.algo_type && DATAFORMAT_NDARRAY == algo_info.input_format &&
        DATAFORMAT_NDARRAY == algo_info.output_format) {
        conv_mgr = new conv2d_ndarray_naive_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::tile_gemm == algo_info.algo_type &&
        DATAFORMAT_NDARRAY == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_ndarray_tile_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::tile_gemm == algo_info.algo_type &&
        DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_tile_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::gemm == algo_info.algo_type && DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::direct_gemm == algo_info.algo_type &&
        DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_direct_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::winograd_b2f3 == algo_info.algo_type &&
        DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_wg_b2f3_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::winograd_b4f3 == algo_info.algo_type &&
        DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_wg_b4f3_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::winograd_b6f3 == algo_info.algo_type &&
        DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_wg_b6f3_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::depthwise == algo_info.algo_type &&
        DATAFORMAT_N4CX == algo_info.input_format &&
        DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_dw_fp32_offline_manager(param, algo_info, allocator);
    }

    return conv_mgr;
}

}}} // namespace ppl::kernel::riscv
