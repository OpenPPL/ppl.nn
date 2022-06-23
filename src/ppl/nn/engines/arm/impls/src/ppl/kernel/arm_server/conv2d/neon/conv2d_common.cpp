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

#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"

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

#include <chrono>
#include <new>
#include <limits>
#include <vector>

#include "ppl/nn/engines/arm/utils/macros.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

conv2d_offline_manager *conv2d_algo_selector::fast_gen_algo(
    const ppl::nn::TensorShape &shape,
    const ppl::nn::arm::EngineOptions &options,
    const ppl::common::isa_t isa_flags,
    const conv2d_param &param,
    ppl::common::Allocator *allocator)
{
#ifdef PPLNN_USE_AARCH64
    ppl::common::datatype_t preferred_data_type = options.forward_precision;
    ppl::common::dataformat_t src_format        = shape.GetDataFormat();

    const bool src_shape_inferred = shape.GetDimCount() >= 4;
    int64_t src_h                 = src_shape_inferred ? shape.GetDim(2) : 0;
    int64_t src_w                 = src_shape_inferred ? shape.GetDim(3) : 0;

    static conv2d_algo_info unknown_info = {
        conv2d_algo::unknown,
        ppl::common::ISA_UNKNOWN,
        ppl::common::DATATYPE_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN};
    (void)unknown_info;

    static conv2d_algo_info fallback_info = {
        conv2d_algo::tile_gemm,
        ppl::common::ISA_ARMV8_2,
        preferred_data_type,
        ppl::common::DATAFORMAT_N8CX,
        ppl::common::DATAFORMAT_N8CX};
    (void)fallback_info;

    if (preferred_data_type == ppl::common::DATATYPE_FLOAT32) {
        if (!(isa_flags & ppl::common::ISA_ARMV8)) {
            LOG(ERROR) << "need armv8 for float32";
            return nullptr;
        }
        if (src_format != ppl::common::DATAFORMAT_NDARRAY &&
            src_format != ppl::common::DATAFORMAT_N4CX) {
            LOG(ERROR) << "need ndarray/n4cx for float32";
            return nullptr;
        }
    } else if (preferred_data_type == ppl::common::DATATYPE_FLOAT16) {
        if (!(isa_flags & ppl::common::ISA_ARMV8_2)) {
            LOG(ERROR) << "need armv8.2 for float16";
            return nullptr;
        }
        if (src_format != ppl::common::DATAFORMAT_NDARRAY &&
            src_format != ppl::common::DATAFORMAT_N8CX) {
            LOG(ERROR) << "need ndarray/n8cx for float16";
            return nullptr;
        }
    } else {
        LOG(ERROR) << "unaccepted data type";
        return nullptr;
    }

    conv2d_algo_info target_algo;
    target_algo.isa       = isa_flags;
    target_algo.data_type = preferred_data_type;
    switch (target_algo.data_type) {
        case ppl::common::DATATYPE_FLOAT16:
            target_algo.output_format = ppl::common::DATAFORMAT_N8CX;
            target_algo.input_format  = (param.channels < 4 && src_format == ppl::common::DATAFORMAT_NDARRAY) ? ppl::common::DATAFORMAT_NDARRAY : ppl::common::DATAFORMAT_N8CX;
            break;

        case ppl::common::DATATYPE_FLOAT32:
            target_algo.output_format = ppl::common::DATAFORMAT_N4CX;
            target_algo.input_format  = (param.channels < 2 && src_format == ppl::common::DATAFORMAT_NDARRAY) ? ppl::common::DATAFORMAT_NDARRAY : ppl::common::DATAFORMAT_N4CX;
            break;

        default:
            LOG(ERROR) << "unaccepted data type";
            return nullptr;
    }

    // check depthwise
    if (param.is_depthwise()) {
        target_algo.algo_type          = ppl::kernel::arm_server::neon::conv2d_algo::depthwise;
        conv2d_offline_manager *dw_mgr = nullptr;
        if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            if (target_algo.input_format == ppl::common::DATAFORMAT_N4CX) {
                dw_mgr = new conv2d_n4cx_depthwise_fp32_offline_manager(param, allocator);
            }
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            if (target_algo.input_format == ppl::common::DATAFORMAT_N8CX) {
                dw_mgr = new conv2d_n8cx_depthwise_fp16_offline_manager(param, allocator);
            }
        }
#endif
        if (dw_mgr != nullptr) {
            if (dw_mgr->is_supported()) {
                dw_mgr->set_algo_info(target_algo);
                dw_mgr->fast_init_schedule_param();
                return dw_mgr;
            } else {
                delete dw_mgr;
            }
        }
    }

    // check direct ndarray
    if (param.group == 1 &&
        target_algo.input_format == ppl::common::DATAFORMAT_NDARRAY) {
        target_algo.algo_type                 = ppl::kernel::arm_server::neon::conv2d_algo::direct_ndarray;
        conv2d_offline_manager *direct_nd_mgr = nullptr;
        if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            direct_nd_mgr = new conv2d_direct_ndarray_fp32_offline_manager(param, allocator);
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            direct_nd_mgr = new conv2d_direct_ndarray_fp16_offline_manager(param, allocator);
        }
#endif
        if (direct_nd_mgr != nullptr) {
            if (direct_nd_mgr->is_supported()) {
                direct_nd_mgr->set_algo_info(target_algo);
                direct_nd_mgr->fast_init_schedule_param();
                return direct_nd_mgr;
            } else {
                delete direct_nd_mgr;
            }
        }
    }

    const bool use_tuning = src_shape_inferred && (options.dynamic_tuning_level != ppl::nn::arm::TUNING_OFF);
    if (use_tuning) {
        return gen_fast_algo(shape, options, isa_flags, param, allocator);
    }

    // check winograd
    if (options.winograd_level != ppl::nn::arm::WG_OFF &&
        param.kernel_h == 3 &&
        param.kernel_w == 3 &&
        param.stride_h == 1 &&
        param.stride_w == 1 &&
        param.dilation_h == 1 &&
        param.dilation_w == 1 &&
        param.channels >= 32 &&
        param.num_output >= 32) {
        switch (options.winograd_level) {
            case ppl::nn::arm::WG_ON:
                if (src_shape_inferred && src_h % 4 == 0 && src_w % 4 == 0) {
                    target_algo.algo_type = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b4f3;
                } else {
                    target_algo.algo_type = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3;
                }
                break;
            case ppl::nn::arm::WG_ON_B2:
                target_algo.algo_type = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3;
                break;
            case ppl::nn::arm::WG_ON_B4:
                target_algo.algo_type = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b4f3;
                break;
            default:
                target_algo.algo_type = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3;
                break;
        }

        conv2d_offline_manager *winograd_mgr = nullptr;
        if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32 &&
            target_algo.input_format == ppl::common::DATAFORMAT_N4CX) {
            switch (target_algo.algo_type) {
                case ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3:
                    winograd_mgr = new conv2d_wgb2f3_fp32_offline_manager(param, allocator);
                    break;

                case ppl::kernel::arm_server::neon::conv2d_algo::winograd_b4f3:
                    winograd_mgr = new conv2d_wgb4f3_fp32_offline_manager(param, allocator);
                    break;

                default:;
            }

        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16 &&
                 target_algo.input_format == ppl::common::DATAFORMAT_N8CX) {
            switch (target_algo.algo_type) {
                case ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3:
                    winograd_mgr = new conv2d_wgb2f3_fp16_offline_manager(param, allocator);
                    break;

                case ppl::kernel::arm_server::neon::conv2d_algo::winograd_b4f3:
                    winograd_mgr = new conv2d_wgb4f3_fp16_offline_manager(param, allocator);
                    break;

                default:;
            }
        }
#endif
        if (winograd_mgr != nullptr) {
            if (winograd_mgr->is_supported()) {
                winograd_mgr->set_algo_info(target_algo);
                winograd_mgr->fast_init_schedule_param();
                return winograd_mgr;
            } else {
                delete winograd_mgr;
            }
        }
    }

    // check im2col
    if (param.kernel_h == 1 && param.stride_h == 1 && param.pad_h == 0 && param.dilation_h == 1 &&
        param.kernel_w == 1 && param.stride_w == 1 && param.pad_w == 0 && param.dilation_w == 1 &&
        target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
        target_algo.algo_type              = ppl::kernel::arm_server::neon::conv2d_algo::tile_gemm;
        conv2d_offline_manager *im2col_mgr = nullptr;
        if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            target_algo.input_format = ppl::common::DATAFORMAT_N4CX;
            im2col_mgr               = new conv2d_n4cx_im2col_fp32_offline_manager(param, allocator);
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            target_algo.input_format = ppl::common::DATAFORMAT_N8CX;
            im2col_mgr               = new conv2d_n8cx_im2col_fp16_offline_manager(param, allocator);
        }
#endif
        if (im2col_mgr != nullptr) {
            if (im2col_mgr->is_supported()) {
                im2col_mgr->set_algo_info(target_algo);
                im2col_mgr->fast_init_schedule_param();
                return im2col_mgr;
            } else {
                delete im2col_mgr;
            }
        }
    }

    // check direct
    target_algo.algo_type              = ppl::kernel::arm_server::neon::conv2d_algo::direct;
    conv2d_offline_manager *direct_mgr = nullptr;
    if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
        if (target_algo.input_format == ppl::common::DATAFORMAT_N4CX) {
            direct_mgr = new conv2d_n4cx_direct_fp32_offline_manager(param, allocator);
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
        if (target_algo.input_format == ppl::common::DATAFORMAT_N8CX) {
            direct_mgr = new conv2d_n8cx_direct_fp16_offline_manager(param, allocator);
        }
    }
#endif
    if (direct_mgr != nullptr) {
        if (direct_mgr->is_supported()) {
            direct_mgr->set_algo_info(target_algo);
            direct_mgr->fast_init_schedule_param();
            return direct_mgr;
        } else {
            delete direct_mgr;
        }
    }
#endif

    return nullptr;
}

static conv2d_offline_manager *get_conv2d_offline_manager_with_algo(
    ppl::kernel::arm_server::neon::conv2d_algo_t algo,
    ppl::common::datatype_t datatype,
    const conv2d_param &param,
    ppl::common::Allocator *allocator)
{
#ifdef PPLNN_USE_AARCH64
    switch (algo) {
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
    }
#endif

    return nullptr;
}

conv2d_offline_manager *conv2d_algo_selector::gen_fast_algo(
    const ppl::nn::TensorShape &src_shape,
    const ppl::nn::arm::EngineOptions &options,
    const ppl::common::isa_t isa_flags,
    const conv2d_param &param,
    ppl::common::Allocator *allocator)
{
#ifdef PPLNN_USE_AARCH64
    ppl::common::datatype_t preferred_data_type = options.forward_precision;
    ppl::common::dataformat_t src_format        = src_shape.GetDataFormat();

    const bool src_shape_inferred = src_shape.GetDimCount() >= 4;

    const int64_t src_h = src_shape_inferred ? src_shape.GetDim(2) : 224;
    const int64_t src_w = src_shape_inferred ? src_shape.GetDim(3) : 224;
    const int64_t dst_h = ((src_h + 2 * param.pad_h - param.dilation_h * (param.kernel_h - 1) - 1) / param.stride_h + 1);
    const int64_t dst_w = ((src_w + 2 * param.pad_w - param.dilation_w * (param.kernel_w - 1) - 1) / param.stride_w + 1);

    static conv2d_algo_info unknown_info = {
        conv2d_algo::unknown,
        ppl::common::ISA_UNKNOWN,
        ppl::common::DATATYPE_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN};
    (void)unknown_info;

    static conv2d_algo_info fallback_info = {
        conv2d_algo::tile_gemm,
        ppl::common::ISA_ARMV8_2,
        preferred_data_type,
        ppl::common::DATAFORMAT_N8CX,
        ppl::common::DATAFORMAT_N8CX};
    (void)fallback_info;

    if (preferred_data_type == ppl::common::DATATYPE_FLOAT32) {
        if (!(isa_flags & ppl::common::ISA_ARMV8)) {
            LOG(ERROR) << "need armv8 for float32";
            return nullptr;
        }
        if (src_format != ppl::common::DATAFORMAT_NDARRAY &&
            src_format != ppl::common::DATAFORMAT_N4CX) {
            LOG(ERROR) << "need ndarray/n4cx for float32";
            return nullptr;
        }
    } else if (preferred_data_type == ppl::common::DATATYPE_FLOAT16) {
        if (!(isa_flags & ppl::common::ISA_ARMV8_2)) {
            LOG(ERROR) << "need armv8.2 for float16";
            return nullptr;
        }
        if (src_format != ppl::common::DATAFORMAT_NDARRAY &&
            src_format != ppl::common::DATAFORMAT_N8CX) {
            LOG(ERROR) << "need ndarray/n8cx for float16";
            return nullptr;
        }
    } else {
        LOG(ERROR) << "unaccepted data type";
        return nullptr;
    }

    conv2d_algo_info target_algo;
    target_algo.isa       = isa_flags;
    target_algo.data_type = preferred_data_type;
    switch (target_algo.data_type) {
        case ppl::common::DATATYPE_FLOAT16:
            target_algo.output_format = ppl::common::DATAFORMAT_N8CX;
            target_algo.input_format  = (param.channels < 4 && src_format == ppl::common::DATAFORMAT_NDARRAY && !param.is_depthwise()) ? ppl::common::DATAFORMAT_NDARRAY : ppl::common::DATAFORMAT_N8CX;
            break;

        case ppl::common::DATATYPE_FLOAT32:
            target_algo.output_format = ppl::common::DATAFORMAT_N4CX;
            target_algo.input_format  = (param.channels < 2 && src_format == ppl::common::DATAFORMAT_NDARRAY && !param.is_depthwise()) ? ppl::common::DATAFORMAT_NDARRAY : ppl::common::DATAFORMAT_N4CX;
            break;

        default:
            LOG(ERROR) << "unaccepted data type";
            return nullptr;
    }

    // no profiling for depthwise
    if (param.is_depthwise()) {
        target_algo.algo_type          = ppl::kernel::arm_server::neon::conv2d_algo::depthwise;
        conv2d_offline_manager *dw_mgr = nullptr;
        if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            if (target_algo.input_format == ppl::common::DATAFORMAT_N4CX) {
                dw_mgr = new conv2d_n4cx_depthwise_fp32_offline_manager(param, allocator);
            }
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            if (target_algo.input_format == ppl::common::DATAFORMAT_N8CX) {
                dw_mgr = new conv2d_n8cx_depthwise_fp16_offline_manager(param, allocator);
            }
        }
#endif
        if (dw_mgr != nullptr) {
            if (dw_mgr->is_supported()) {
                dw_mgr->set_algo_info(target_algo);
                dw_mgr->fast_init_schedule_param();
                return dw_mgr;
            } else {
                delete dw_mgr;
            }
        }
    }

    // no profiling for ndarray
    if (target_algo.input_format == ppl::common::DATAFORMAT_NDARRAY) {
        target_algo.algo_type                      = ppl::kernel::arm_server::neon::conv2d_algo::direct_ndarray;
        conv2d_offline_manager *direct_ndarray_mgr = nullptr;
        if (target_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            direct_ndarray_mgr = new conv2d_direct_ndarray_fp32_offline_manager(param, allocator);
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (target_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            direct_ndarray_mgr = new conv2d_direct_ndarray_fp16_offline_manager(param, allocator);
        }
#endif
        if (direct_ndarray_mgr != nullptr) {
            if (direct_ndarray_mgr->is_supported()) {
                direct_ndarray_mgr->set_algo_info(target_algo);
                direct_ndarray_mgr->fast_init_schedule_param();
                return direct_ndarray_mgr;
            } else {
                delete direct_ndarray_mgr;
            }
        }
    }

    std::vector<ppl::kernel::arm_server::neon::conv2d_algo_t> candidate_algo_list;
    if (param.kernel_h == 3 &&
        param.kernel_w == 3 &&
        param.stride_h == 1 &&
        param.stride_w == 1 &&
        param.dilation_h == 1 &&
        param.dilation_w == 1) {
        auto algo = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b2f3;
        candidate_algo_list.push_back(algo);
        algo = ppl::kernel::arm_server::neon::conv2d_algo::winograd_b4f3;
        candidate_algo_list.push_back(algo);
    }
    auto algo = ppl::kernel::arm_server::neon::conv2d_algo::direct;
    candidate_algo_list.push_back(algo);
    algo = ppl::kernel::arm_server::neon::conv2d_algo::tile_gemm;
    candidate_algo_list.push_back(algo);

    const bool tune_sp   = true;
    double best_run_time = std::numeric_limits<double>::max();

    uint32_t elem_size = ppl::common::GetSizeOfDataType(target_algo.data_type);
    uint32_t num_lanes = 16 / elem_size;
    const uint32_t pad_channels = ((param.channels   + (num_lanes - 1)) / num_lanes) * num_lanes;
    const uint32_t pad_num_outs = ((param.num_output + (num_lanes - 1)) / num_lanes) * num_lanes;

    const int64_t num_batch = src_shape.GetDim(0);
    ppl::nn::TensorShape dst_shape;
    dst_shape.Reshape({num_batch, pad_num_outs, dst_h, dst_w});

    const size_t src_size  = num_batch * pad_channels * src_h * src_w * elem_size;
    const size_t dst_size  = num_batch * pad_num_outs * dst_h * dst_w * elem_size;
    const size_t bias_size = pad_num_outs * elem_size;

    void *src = allocator->Alloc(src_size);
    void *dst = allocator->Alloc(dst_size);
    void *bias = allocator->Alloc(bias_size);

    ppl::kernel::arm_server::neon::conv2d_algo_t best_algo = ppl::kernel::arm_server::neon::conv2d_algo::unknown;
    conv2d_offline_manager *best_conv2d_mgr                = nullptr;
    for (auto candidate_algo : candidate_algo_list) {
        conv2d_offline_manager *conv2d_mgr = get_conv2d_offline_manager_with_algo(candidate_algo, target_algo.data_type, param, allocator);
        if (conv2d_mgr == nullptr) {
            continue;
        }
        if (!conv2d_mgr->is_supported()) {
            delete conv2d_mgr;
            continue;
        }

        double run_time = std::numeric_limits<double>::max();
        conv2d_mgr->pick_best_schedule_param(src_shape, src, bias, dst_shape, dst, tune_sp, run_time);
        if (run_time <= best_run_time) {
            best_algo     = candidate_algo;
            best_run_time = run_time;

            if (best_conv2d_mgr) {
                delete best_conv2d_mgr;
            }
            best_conv2d_mgr = conv2d_mgr;

            target_algo.algo_type = best_algo;
            best_conv2d_mgr->set_algo_info(target_algo);
        } else {
            delete conv2d_mgr;
        }
    }

    allocator->Free(src);
    allocator->Free(dst);
    allocator->Free(bias);

    LOG(DEBUG) << "Selected conv2d algorithm: " << best_algo;
    return best_conv2d_mgr;
#else
    return nullptr;
#endif
}

}}}}; // namespace ppl::kernel::arm_server::neon
