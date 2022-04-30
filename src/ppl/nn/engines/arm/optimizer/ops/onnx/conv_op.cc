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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/conv_op.h"

#include <cstring>

#include "ppl/nn/engines/arm/kernels/onnx/conv2d_kernel.h"
#include "ppl/nn/engines/arm/utils/data_trans.h"
#include "ppl/nn/oputils/onnx/reshape_conv.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ConvOp::~ConvOp() {
    if (conv2d_param_ != nullptr) {
        if (conv2d_param_->mgr != nullptr) {
            conv2d_param_->mgr->release_cvt_weights();
            delete conv2d_param_->mgr;
        }
        if (conv2d_param_->fallback_mgr != nullptr) {
            conv2d_param_->fallback_mgr->release_cvt_weights();
            delete conv2d_param_->fallback_mgr;
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

    float* weight_data = (float*)weight_data_it->second.data.data();
    int64_t weight_len = weight_data_it->second.data.size() / sizeof(float);

    float* bias_data = nullptr;
    int64_t bias_len = 0;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it == graph_data->constants.end()) {
            LOG(INFO) << "ConvOp constant weight not found, will use conv runtime.";
            return ppl::common::RC_SUCCESS;
        }
        bias_data = (float*)bias_data_it->second.data.data();
        bias_len = bias_data_it->second.data.size() / sizeof(float);
    }

    const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
    const int64_t kernel_dims = weight_shape.dims.size() - 2;

    // Check Param
    const ppl::nn::onnx::ConvParam& conv_param = *param_;
    for (int64_t i = 0; i < kernel_dims; ++i) {
        if (conv_param.pads[i] != conv_param.pads[i + kernel_dims]) {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    if (kernel_dims == 2) {
        if (!conv2d_param_) {
            conv2d_param_ = new Convolution2DParam;
        }
        if (!conv2d_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        const int32_t num_output = weight_shape.dims[0];
        const int32_t channels = weight_shape.dims[1] * param_->group;

        ppl::kernel::arm_server::neon::conv2d_param& conv2d_kernel_param = conv2d_param_->param;
        conv2d_kernel_param.kernel_h = conv_param.kernel_shape[0];
        conv2d_kernel_param.kernel_w = conv_param.kernel_shape[1];
        conv2d_kernel_param.stride_h = conv_param.strides[0];
        conv2d_kernel_param.stride_w = conv_param.strides[1];
        conv2d_kernel_param.pad_h = conv_param.pads[0];
        conv2d_kernel_param.pad_w = conv_param.pads[1];
        conv2d_kernel_param.dilation_h = conv_param.dilations[0];
        conv2d_kernel_param.dilation_w = conv_param.dilations[1];
        conv2d_kernel_param.group = conv_param.group;
        conv2d_kernel_param.num_output = num_output;
        conv2d_kernel_param.channels = channels;
        conv2d_kernel_param.fuse_flag = 0;

        conv2d_param_->mgr = ppl::kernel::arm_server::neon::conv2d_algo_selector::fast_gen_algo(
            *info.GetInput<TensorImpl>(0)->GetShape(), *options.engine_options, options.device->GetISA(),
            conv2d_param_->param, options.device->GetAllocator());

        if (conv2d_param_->mgr == nullptr) {
            LOG(ERROR) << "No algorithm selected.";
            return ppl::common::RC_UNSUPPORTED;
        }

        auto selected_algo = conv2d_param_->mgr->algo_info();
        if (selected_algo.algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
            LOG(ERROR) << "Unsupported algorithm type: " << selected_algo.algo_type;
            return ppl::common::RC_UNSUPPORTED;
        }
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        LOG(INFO) << "Op " << node->GetName() << " selected conv algorithm: "
                  << ppl::kernel::arm_server::neon::get_conv_algo_str(selected_algo.algo_type);
#endif

        ppl::common::RetCode normal_cvt_weights_ret = ppl::common::RC_SUCCESS;
        ppl::common::RetCode fallback_cvt_weights_ret = ppl::common::RC_SUCCESS;
        if (selected_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            if (bias_data != nullptr) {
                normal_cvt_weights_ret = conv2d_param_->mgr->gen_cvt_weights(weight_data, bias_data);

                if (conv2d_param_->fallback_mgr) {
                    fallback_cvt_weights_ret = conv2d_param_->fallback_mgr->gen_cvt_weights(weight_data, bias_data);
                }
            } else {
                std::vector<float> zero_bias(conv2d_kernel_param.num_output, 0.0f);
                normal_cvt_weights_ret = conv2d_param_->mgr->gen_cvt_weights(weight_data, zero_bias.data());

                if (conv2d_param_->fallback_mgr) {
                    fallback_cvt_weights_ret =
                        conv2d_param_->fallback_mgr->gen_cvt_weights(weight_data, zero_bias.data());
                }
            }
        } else if (selected_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            vector<__fp16> weight_data_fp16;
            weight_data_fp16.resize(weight_len * sizeof(__fp16));
            Fp32ToFp16(weight_data, weight_len, weight_data_fp16.data());

            if (bias_data != nullptr) {
                vector<__fp16> bias_data_fp16;
                bias_data_fp16.resize(bias_len * sizeof(__fp16));
                Fp32ToFp16(bias_data, bias_len, bias_data_fp16.data());
                normal_cvt_weights_ret =
                    conv2d_param_->mgr->gen_cvt_weights(weight_data_fp16.data(), bias_data_fp16.data());

                if (conv2d_param_->fallback_mgr) {
                    fallback_cvt_weights_ret =
                        conv2d_param_->fallback_mgr->gen_cvt_weights(weight_data_fp16.data(), bias_data_fp16.data());
                }
            } else {
                std::vector<__fp16> zero_bias(conv2d_kernel_param.num_output, 0.0f);
                normal_cvt_weights_ret = conv2d_param_->mgr->gen_cvt_weights(weight_data_fp16.data(), zero_bias.data());

                if (conv2d_param_->fallback_mgr) {
                    fallback_cvt_weights_ret =
                        conv2d_param_->fallback_mgr->gen_cvt_weights(weight_data_fp16.data(), zero_bias.data());
                }
            }
        } else {
            LOG(ERROR) << "Unsupported data type: " << selected_algo.data_type;
            return ppl::common::RC_UNSUPPORTED;
        }
        if (ppl::common::RC_SUCCESS != normal_cvt_weights_ret || ppl::common::RC_SUCCESS != fallback_cvt_weights_ret) {
            LOG(ERROR) << "algo " << selected_algo.algo_type << " cvt weights failed.";
        }
    } else {
        LOG(ERROR) << "Unsupported kernel dim: " << kernel_dims;
        return ppl::common::RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

RetCode ConvOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (conv2d_param_ && conv2d_param_->mgr &&
        conv2d_param_->mgr->algo_info().algo_type != ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        selected_input_formats->at(0) = conv2d_param_->mgr->algo_info().input_format;
        selected_output_formats->at(0) = conv2d_param_->mgr->algo_info().output_format;
        return RC_SUCCESS;
    }
    return RC_INVALID_VALUE;
}
RetCode ConvOp::SelectDataType(const InputOutputInfo& info, std::vector<ppl::common::datatype_t>* selected_input_types,
                               std::vector<ppl::common::datatype_t>* selected_output_types,
                               const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    for (uint32_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_types->at(i) = info.GetInput<TensorImpl>(i)->GetShape()->GetDataType();
    }
    return RC_SUCCESS;
}

bool ConvOp::TryFuseReLU(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        return false;
    }
    ppl::kernel::arm_server::neon::conv2d_param param = conv2d_param_->mgr->param();
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::RELU;
    conv2d_param_->mgr->set_param(param);
    return true;
}

bool ConvOp::TryFuseReLU6(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        return false;
    }
    ppl::kernel::arm_server::neon::conv2d_param param = conv2d_param_->mgr->param();
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::RELU;
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::RELU6;
    conv2d_param_->mgr->set_param(param);
    return true;
}

bool ConvOp::TryFuseSum(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        return false;
    }
    ppl::kernel::arm_server::neon::conv2d_param param = conv2d_param_->mgr->param();
    if (param.fuse_flag) { // already fused sum, relu or relu6
        return false;
    }
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::SUM;
    conv2d_param_->mgr->set_param(param);
    return true;
}

KernelImpl* ConvOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<Conv2dKernel>(conv2d_param_);
}

}}} // namespace ppl::nn::arm
