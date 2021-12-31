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

#include "ppl/kernel/riscv/fp16/conv2d/tile_gemm/vec128/conv2d_n8cx_tile_gemm_fp16_vec128.h"
#include "ppl/kernel/riscv/fp16/conv2d/tile_gemm/vec128/conv2d_n8cx_tile_gemm_cto8c_fp16_vec128.h"
#include "ppl/kernel/riscv/fp16/conv2d/wg/vec128/conv2d_n8cx_wg_b2f3_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/wg/vec128/conv2d_n8cx_wg_b4f3_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/wg/vec128/conv2d_n8cx_wg_b6f3_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/depthwise/vec128/conv2d_n8cx_dw_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/naive/conv2d_ndarray_naive_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d.h"
#include "ppl/common/log.h"
#include "ppl/common/types.h"

namespace ppl { namespace kernel { namespace riscv {

conv2d_common_algo_info conv2d_fp16_algo_selector::select_algo(const ppl::nn::TensorShape& input_shape,
                                                               const conv2d_common_param& param) {
    static conv2d_common_algo_info unknown_info = {conv2d_common_algo::unknown, ppl::common::DATAFORMAT_UNKNOWN,
                                                   ppl::common::DATAFORMAT_UNKNOWN, ppl::common::DATATYPE_FLOAT16,
                                                   ppl::common::DATATYPE_FLOAT16};

    if (param.dilation_h != 1 || param.dilation_w != 1) {
        return unknown_info;
    }

    if (input_shape.GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (param.group == 1) {
            return {conv2d_common_algo::tile_gemm, ppl::common::DATAFORMAT_NDARRAY, ppl::common::DATAFORMAT_N8CX,
                    ppl::common::DATATYPE_FLOAT16, ppl::common::DATATYPE_FLOAT16};
        }
    }

    if (input_shape.GetDataFormat() == ppl::common::DATAFORMAT_N8CX) {
        if (param.group == param.num_output && param.num_output == param.channels) {
            return {conv2d_common_algo::depthwise, ppl::common::DATAFORMAT_N8CX, ppl::common::DATAFORMAT_N8CX,
                    ppl::common::DATATYPE_FLOAT16, ppl::common::DATATYPE_FLOAT16};
        } else {
            if (param.kernel_h == 3 && param.kernel_w == 3 && param.stride_h == 1 && param.stride_w == 1) {
                return {conv2d_common_algo::winograd_b4f3, ppl::common::DATAFORMAT_N8CX, ppl::common::DATAFORMAT_N8CX,
                        ppl::common::DATATYPE_FLOAT16, ppl::common::DATATYPE_FLOAT16};
            } else {
                return {conv2d_common_algo::tile_gemm, ppl::common::DATAFORMAT_N8CX, ppl::common::DATAFORMAT_N8CX,
                        ppl::common::DATATYPE_FLOAT16, ppl::common::DATATYPE_FLOAT16};
            }
        }
    } else {
        return {conv2d_common_algo::tile_gemm, ppl::common::DATAFORMAT_N8CX, ppl::common::DATAFORMAT_N8CX,
                ppl::common::DATATYPE_FLOAT16, ppl::common::DATATYPE_FLOAT16};
    }

    return unknown_info;
}

conv2d_offline_manager<__fp16>* conv2d_fp16_algo_selector::gen_algo(const conv2d_common_param& param,
                                                                    const conv2d_common_algo_info& algo_info,
                                                                    ppl::common::Allocator* allocator) {
    conv2d_offline_manager<__fp16>* conv_mgr = nullptr;

    if (algo_info.algo_type == conv2d_common_algo::tile_gemm &&
        algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
        algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_tile_gemm_fp16_offline_manager(param, algo_info, allocator);

    } else if (algo_info.algo_type == conv2d_common_algo::tile_gemm &&
               algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
               algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_tile_gemm_cto8c_fp16_offline_manager(param, algo_info, allocator);

    } else if (algo_info.algo_type == conv2d_common_algo::naive &&
               algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY &&
               algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_ndarray_naive_fp16_offline_manager(param, algo_info, allocator);

    } else if (algo_info.algo_type == conv2d_common_algo::winograd_b2f3 &&
               algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
               algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_wg_b2f3_fp16_offline_manager(param, algo_info, allocator);
    } else if (algo_info.algo_type == conv2d_common_algo::winograd_b4f3 &&
               algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
               algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_wg_b4f3_fp16_offline_manager(param, algo_info, allocator);
    } else if (algo_info.algo_type == conv2d_common_algo::winograd_b6f3 &&
               algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
               algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_wg_b6f3_fp16_offline_manager(param, algo_info, allocator);
    } else if (algo_info.algo_type == conv2d_common_algo::depthwise &&
               algo_info.input_format == ppl::common::DATAFORMAT_N8CX &&
               algo_info.output_format == ppl::common::DATAFORMAT_N8CX) {
        conv_mgr = new conv2d_n8cx_dw_fp16_offline_manager(param, algo_info, allocator);
    }

    return conv_mgr;
}

}}}; // namespace ppl::kernel::riscv
