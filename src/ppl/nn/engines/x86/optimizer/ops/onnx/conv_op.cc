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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/conv_kernel.h"
#include "ppl/nn/engines/x86/kernels/onnx/conv2d_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_conv.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/common/threading_tools.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

ConvOp::~ConvOp() {
    if (conv2d_param_ != nullptr) {
        if (conv2d_param_->mgr != nullptr) {
            conv2d_param_->mgr->release_cvt_weights();
        }
        if (conv2d_param_->fallback_mgr != nullptr) {
            conv2d_param_->fallback_mgr->release_cvt_weights();
        }
        delete conv2d_param_;
    }
}

RetCode ConvOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph_data;
    const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;

    const int64_t kernel_dims =
        param_->kernel_shape.size() == 0 ? (weight_shape.dims.size() - 2) : param_->kernel_shape.size();

    if (kernel_dims != 2) {
        LOG(ERROR) << "Only support Conv2d currently. Get unsupported kernel_dims=" << kernel_dims << ", which is Conv("
                   << kernel_dims << "d)";
        return ppl::common::RC_UNSUPPORTED;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConv(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

ppl::common::RetCode ConvOp::SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    if (weight_data_it == graph_data->constants.end()) {
        LOG(INFO) << "ConvOp constant weight not found, will use conv runtime.";
        return ppl::common::RC_SUCCESS;
    }

    const float* weight_data = (const float*)weight_data_it->second.data.GetData();
    const float* bias_data = nullptr;

    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it == graph_data->constants.end()) {
            LOG(INFO) << "ConvOp constant weight not found, will use conv runtime.";
            return ppl::common::RC_SUCCESS;
        }
        bias_data = (const float*)bias_data_it->second.data.GetData();
    }

    const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;

    bias_term_ = (node->GetInputCount() == 3) ? 1 : 0;

    // Check Param
    const ppl::nn::onnx::ConvParam& conv_param = *param_;
    const int64_t kernel_dims =
        param_->kernel_shape.size() == 0 ? (weight_shape.dims.size() - 2) : param_->kernel_shape.size();

    for (int64_t i = 0; i < kernel_dims; ++i) {
        if (conv_param.pads[i] != conv_param.pads[i + kernel_dims]) {
            LOG(ERROR) << "only support symmetrical pads now.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    if (kernel_dims == 2) {
        if (!conv2d_param_) {
            conv2d_param_ = new Conv2dParam;
        }
        if (!conv2d_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        const int32_t num_output = weight_shape.dims[0];
        const int32_t channels = weight_shape.dims[1] * param_->group;

        ppl::kernel::x86::conv2d_fp32_param& conv2d_param = conv2d_param_->param;
        conv2d_param.kernel_h = conv_param.kernel_shape[0];
        conv2d_param.kernel_w = conv_param.kernel_shape[1];
        conv2d_param.stride_h = conv_param.strides[0];
        conv2d_param.stride_w = conv_param.strides[1];
        conv2d_param.pad_h = conv_param.pads[0];
        conv2d_param.pad_w = conv_param.pads[1];
        conv2d_param.dilation_h = conv_param.dilations[0];
        conv2d_param.dilation_w = conv_param.dilations[1];
        conv2d_param.group = conv_param.group;
        conv2d_param.num_output = num_output;
        conv2d_param.channels = channels;
        conv2d_param.fuse_flag = 0;

        conv2d_param_->algo_info = ppl::kernel::x86::conv2d_algo_selector::select_algo(
            info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat(), conv2d_param_->param, options.device->GetISA());

        if (conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
            LOG(INFO) << "Conv select algorithm failed, use fallback kernel";
        } else {
            conv2d_param_->mgr = ppl::kernel::x86::conv2d_algo_selector::gen_algo(
                conv2d_param_->param, conv2d_param_->algo_info, options.device->GetAllocator());

            // winograd b4f3 avx512 may fallback to direct
            if (conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::WINOGRAD_B4F3) {
                conv2d_param_->algo_info.algo_type = ppl::kernel::x86::conv2d_fp32_algo::DIRECT;
                conv2d_param_->fallback_mgr = ppl::kernel::x86::conv2d_algo_selector::gen_algo(
                    conv2d_param_->param, conv2d_param_->algo_info, options.device->GetAllocator());
                conv2d_param_->infer_fallback_func = [](const TensorImpl* X, const TensorImpl* Y,
                                                        const ppl::kernel::x86::conv2d_fp32_param* param) -> bool {
                    const int64_t dst_h = Y->GetShape()->GetDim(2);
                    const int64_t dst_w = Y->GetShape()->GetDim(3);
                    const int64_t batch = X->GetShape()->GetDim(0);
                    const int64_t num_tiles = batch * ((dst_h + 3) / 4) * ((dst_w + 3) / 4);
                    const bool align_tiles = (dst_h % 4 == 0) && (dst_w % 4) == 0;

                    const int64_t num_threads = ppl::kernel::x86::get_omp_max_threads();
                    if (num_threads > 4) { // Maybe memory bound. Just maybe.
                        if (param->group > 4) {
                            if (param->channels / param->group <= 2 * 1.801f * 16) { // Multigroup need more channels
                                return true;
                            }
                        }
                        if (param->group / num_threads > 1 && num_threads / batch <= 4) { // Many group but small batch
                            return true;
                        }
                    }
                    return num_tiles < (align_tiles ? 10 : 12);
                };
                conv2d_param_->algo_info.algo_type = ppl::kernel::x86::conv2d_fp32_algo::WINOGRAD_B4F3;
            }

            if (bias_data != nullptr) {
                conv2d_param_->mgr->gen_cvt_weights(weight_data, bias_data);
                if (conv2d_param_->fallback_mgr) {
                    conv2d_param_->fallback_mgr->gen_cvt_weights(weight_data, bias_data);
                }
            } else {
                std::vector<float> zero_bias(conv2d_param.num_output, 0.0f);
                conv2d_param_->mgr->gen_cvt_weights(weight_data, zero_bias.data());
                if (conv2d_param_->fallback_mgr) {
                    conv2d_param_->fallback_mgr->gen_cvt_weights(weight_data, zero_bias.data());
                }
            }
        }
    } else {
        LOG(ERROR) << "Unsupported kernel dim: " << kernel_dims;
        return ppl::common::RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

RetCode ConvOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (conv2d_param_ && conv2d_param_->algo_info.algo_type != ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        selected_input_formats->at(0) = conv2d_param_->algo_info.input_format;
        if (conv2d_param_->mgr->param().fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM) {
            selected_input_formats->at(info.GetInputCount() - 1) = conv2d_param_->algo_info.input_format;
        }
        selected_output_formats->at(0) = conv2d_param_->algo_info.output_format;
    }
    return RC_SUCCESS;
}

RetCode ConvOp::OmitConstantsData(std::map<edgeid_t, int64_t>* constants_data_refcount) {
    if (conv2d_param_ && conv2d_param_->algo_info.algo_type != ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        auto weight_id = GetNode()->GetInput(1);
        auto it = constants_data_refcount->find(weight_id);
        if (it != constants_data_refcount->end()) {
            it->second--;
        }
        if (bias_term_) {
            auto bias_id = GetNode()->GetInput(2);
            it = constants_data_refcount->find(bias_id);
            if (it != constants_data_refcount->end()) {
                it->second--;
            }
        }
    }
    return RC_SUCCESS;
}

bool ConvOp::TryFuseReLU() {
    if (!conv2d_param_ || conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        return false;
    }
    ppl::kernel::x86::conv2d_fp32_param param = conv2d_param_->mgr->param();
    param.fuse_flag |= ppl::kernel::x86::conv_fuse_flag::RELU;
    conv2d_param_->mgr->set_param(param);
    if (conv2d_param_->fallback_mgr) {
        conv2d_param_->fallback_mgr->set_param(param);
    }
    return true;
}

bool ConvOp::TryFuseReLU6() {
    if (!conv2d_param_ || conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        return false;
    }
    ppl::kernel::x86::conv2d_fp32_param param = conv2d_param_->mgr->param();
    param.fuse_flag |= ppl::kernel::x86::conv_fuse_flag::RELU6;
    conv2d_param_->mgr->set_param(param);
    if (conv2d_param_->fallback_mgr) {
        conv2d_param_->fallback_mgr->set_param(param);
    }
    return true;
}

bool ConvOp::TryFuseSum() {
    if (!conv2d_param_ || conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        return false;
    }
    ppl::kernel::x86::conv2d_fp32_param param = conv2d_param_->mgr->param();
    if ((param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::RELU) || // sum cannot fuse behind activation
        (param.fuse_flag & ppl::kernel::x86::conv_fuse_flag::RELU6)) {
        return false;
    }
    param.fuse_flag |= ppl::kernel::x86::conv_fuse_flag::SUM;
    conv2d_param_->mgr->set_param(param);
    if (conv2d_param_->fallback_mgr) {
        conv2d_param_->fallback_mgr->set_param(param);
    }
    return true;
}

KernelImpl* ConvOp::CreateKernelImpl() const {
    if (!conv2d_param_ || conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        return CreateKernelImplWithParam<ConvKernel>(param_.get());
    }
    return CreateKernelImplWithParam<Conv2dKernel>(conv2d_param_);
}

}}} // namespace ppl::nn::x86
