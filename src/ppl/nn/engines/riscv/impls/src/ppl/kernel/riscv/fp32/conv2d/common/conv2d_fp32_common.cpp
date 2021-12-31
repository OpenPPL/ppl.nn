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

namespace ppl { namespace kernel { namespace riscv {

conv2d_common_algo_info conv2d_fp32_algo_selector::select_algo(const ppl::nn::TensorShape& input_shape,
                                                               const conv2d_common_param& param) {
    LOG(DEBUG) << "RISCV FP32 CONV select algo";
    static conv2d_common_algo_info unknown_info = {conv2d_common_algo::unknown, ppl::common::DATAFORMAT_UNKNOWN,
                                                   ppl::common::DATAFORMAT_UNKNOWN, ppl::common::DATATYPE_FLOAT32,
                                                   ppl::common::DATATYPE_FLOAT32};

    if (param.dilation_h != 1 || param.dilation_w != 1) {
        return unknown_info;
    }

    if (ppl::common::DATAFORMAT_NDARRAY == input_shape.GetDataFormat()) {
        if (param.group == 1) {
            return {conv2d_common_algo::tile_gemm, ppl::common::DATAFORMAT_NDARRAY, ppl::common::DATAFORMAT_N4CX,
                    ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_FLOAT32};
        }
    }

    if (ppl::common::DATAFORMAT_N4CX == input_shape.GetDataFormat()) {
        if (param.group == 1 && param.kernel_h == 1 && param.kernel_w == 1 && param.pad_h == 0 && param.pad_w == 0 &&
            param.stride_h == 1 && param.stride_w == 1 && param.dilation_h == 1 && param.dilation_w == 1) {
            return {conv2d_common_algo::gemm, ppl::common::DATAFORMAT_N4CX, ppl::common::DATAFORMAT_N4CX,
                    ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_FLOAT32};
        }
        if (param.group == param.num_output && param.num_output == param.channels) {
            return {conv2d_common_algo::depthwise, ppl::common::DATAFORMAT_N4CX, ppl::common::DATAFORMAT_N4CX,
                    ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_FLOAT32};
        } else {
            if (param.kernel_h == 3 && param.kernel_w == 3 && param.stride_h == 1 && param.stride_w == 1) {
                return {// conv2d_common_algo::winograd_b2f3,
                        conv2d_common_algo::winograd_b4f3,
                        // conv2d_common_algo::winograd_b6f3,
                        ppl::common::DATAFORMAT_N4CX, ppl::common::DATAFORMAT_N4CX, ppl::common::DATATYPE_FLOAT32,
                        ppl::common::DATATYPE_FLOAT32};
            } else {
                return {conv2d_common_algo::tile_gemm, ppl::common::DATAFORMAT_N4CX, ppl::common::DATAFORMAT_N4CX,
                        ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_FLOAT32};
            }
        }
    } else {
        // if (param.group == 1 &&
        //     param.kernel_h == 1 && param.kernel_w == 1 &&
        //     param.stride_h == 1 && param.stride_w == 1 &&
        //     param.pad_h == 0 && param.pad_w == 0) {
        //     return {
        //         conv2d_common_algo::direct_gemm,
        //         ppl::common::DATAFORMAT_N4CX,
        //         ppl::common::DATAFORMAT_N4CX,
        //         ppl::common::DATATYPE_FLOAT32,
        //         ppl::common::DATATYPE_FLOAT32
        //     };
        // }

        return {conv2d_common_algo::tile_gemm, ppl::common::DATAFORMAT_N4CX, ppl::common::DATAFORMAT_N4CX,
                ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_FLOAT32};
    }

    // return {
    //     conv2d_common_algo::tile_gemm,
    //     ppl::common::DATAFORMAT_N4CX,
    //     ppl::common::DATAFORMAT_N4CX,
    //     ppl::common::DATATYPE_FLOAT32,
    //     ppl::common::DATATYPE_FLOAT32
    // };

    return unknown_info;
}

conv2d_offline_manager<float>* conv2d_fp32_algo_selector::gen_algo(const conv2d_common_param& param,
                                                                   const conv2d_common_algo_info& algo_info,
                                                                   ppl::common::Allocator* allocator) {
    LOG(DEBUG) << "RISCV FP32 CONV gen algo";
    conv2d_offline_manager<float>* conv_mgr = nullptr;

    if (conv2d_common_algo::naive == algo_info.algo_type && ppl::common::DATAFORMAT_NDARRAY == algo_info.input_format &&
        ppl::common::DATAFORMAT_NDARRAY == algo_info.output_format) {
        conv_mgr = new conv2d_ndarray_naive_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::tile_gemm == algo_info.algo_type &&
        ppl::common::DATAFORMAT_NDARRAY == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_ndarray_tile_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::tile_gemm == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_tile_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::gemm == algo_info.algo_type && ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::direct_gemm == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_direct_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::winograd_b2f3 == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_wg_b2f3_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::winograd_b4f3 == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_wg_b4f3_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::winograd_b6f3 == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_wg_b6f3_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::depthwise == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        conv_mgr = new conv2d_n4cx_dw_fp32_offline_manager(param, algo_info, allocator);
    }

    return conv_mgr;
}

}}}; // namespace ppl::kernel::riscv
