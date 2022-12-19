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

#include "ppl/kernel/arm_server/common/internal_include.h"
#ifdef PPLNN_USE_AARCH64
#include "ppl/kernel/arm_server/conv2d/neon/fp16/depthwise/conv2d_n8cx_depthwise_fp16.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct/conv2d_n8cx_direct_fp16.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct_ndarray/conv2d_direct_ndarray_fp16.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/im2col/conv2d_n8cx_im2col_fp16.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/winograd/conv2d_wgb2f3_fp16.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/winograd/conv2d_wgb4f3_fp16.h"

#include "ppl/kernel/arm_server/conv2d/neon/fp32/depthwise/conv2d_n4cx_depthwise_fp32.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/direct/conv2d_n4cx_direct_fp32.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/direct_ndarray/conv2d_direct_ndarray_fp32.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/im2col/conv2d_n4cx_im2col_fp32.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/winograd/conv2d_wgb2f3_fp32.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp32/winograd/conv2d_wgb4f3_fp32.h"
#endif
#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

conv2d_offline_manager * conv2d_algo::generate_conv_mgr(
    const ppl::kernel::arm_server::neon::conv2d_algo_t algo,
    const ppl::common::datatype_t datatype,
    const conv2d_param &param,
    ppl::common::Allocator *allocator)
{
#ifdef PPLNN_USE_AARCH64
    switch (algo) {
        case ppl::kernel::arm_server::neon::conv2d_algo::depthwise:
            if (datatype == ppl::common::DATATYPE_FLOAT32) {
                return new conv2d_n4cx_depthwise_fp32_offline_manager(param, allocator);
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
            else if (datatype == ppl::common::DATATYPE_FLOAT16) {
                return new conv2d_n8cx_depthwise_fp16_offline_manager(param, allocator);
            }
#endif
            break;

        case ppl::kernel::arm_server::neon::conv2d_algo::direct:
            if (datatype == ppl::common::DATATYPE_FLOAT32) {
                return new conv2d_n4cx_direct_fp32_offline_manager(param, allocator);
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
            else if (datatype == ppl::common::DATATYPE_FLOAT16) {
                return new conv2d_n8cx_direct_fp16_offline_manager(param, allocator);
            }
#endif
            break;
        
        case ppl::kernel::arm_server::neon::conv2d_algo::direct_ndarray:
            if (datatype == ppl::common::DATATYPE_FLOAT32) {
                return new conv2d_direct_ndarray_fp32_offline_manager(param, allocator);
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
            else if (datatype == ppl::common::DATATYPE_FLOAT16) {
                return new conv2d_direct_ndarray_fp16_offline_manager(param, allocator);
            }
#endif
            break;

        case ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3:
            if (datatype == ppl::common::DATATYPE_FLOAT32) {
                return new conv2d_wgb2f3_fp32_offline_manager(param, allocator);
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
            else if (datatype == ppl::common::DATATYPE_FLOAT16) {
                return new conv2d_wgb2f3_fp16_offline_manager(param, allocator);
            }
#endif
            break;

        case ppl::kernel::arm_server::neon::conv2d_algo::winograd_b4f3:
            if (datatype == ppl::common::DATATYPE_FLOAT32) {
                return new conv2d_wgb4f3_fp32_offline_manager(param, allocator);
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
            else if (datatype == ppl::common::DATATYPE_FLOAT16) {
                return new conv2d_wgb4f3_fp16_offline_manager(param, allocator);
            }
#endif
            break;

        case ppl::kernel::arm_server::neon::conv2d_algo::tile_gemm:
            if (datatype == ppl::common::DATATYPE_FLOAT32) {
                return new conv2d_n4cx_im2col_fp32_offline_manager(param, allocator);
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
            else if (datatype == ppl::common::DATATYPE_FLOAT16) {
                return new conv2d_n8cx_im2col_fp16_offline_manager(param, allocator);
            }
#endif
            break;

        default:
            return nullptr;
    }
#endif

    return nullptr;
}


}}}}