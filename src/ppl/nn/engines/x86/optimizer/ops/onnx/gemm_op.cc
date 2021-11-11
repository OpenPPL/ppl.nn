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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/engines/x86/kernels/onnx/fc_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

GemmOp::~GemmOp() {
    if (fc_param_ != nullptr) {
        if (fc_param_->mgr != nullptr) {
            fc_param_->mgr->release_cvt_weights();
            delete fc_param_->mgr;
        }
        delete fc_param_;
    }
}

RetCode GemmOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    const float* weight_data = nullptr;
    if (weight_data_it != graph_data->constants.end()) {
        weight_data = (const float*)weight_data_it->second.data.data();
    }

    const float* bias_data = nullptr;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it != graph_data->constants.end()) {
            bias_data = (const float*)bias_data_it->second.data.data();
        }
    }

    if (!param_->transA && param_->transB && weight_data != nullptr) {
        if (!fc_param_) {
            fc_param_ = new FCParam;
        }
        if (!fc_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
        fc_param_->param.num_output = weight_shape.dims[0];
        fc_param_->param.channels = weight_shape.dims[1];
        fc_param_->param.fuse_flag = 0;

        fc_param_->algo_info = ppl::kernel::x86::fc_algo_selector::select_algo(
            ppl::common::DATAFORMAT_NDARRAY, fc_param_->param, options.device->GetISA());
        if (fc_param_->algo_info.algo_type == ppl::kernel::x86::fc_fp32_algo::UNKNOWN) {
            LOG(INFO) << "FC select algorithm failed, use fallback kernel";
        } else {
            fc_param_->mgr = ppl::kernel::x86::fc_algo_selector::gen_algo(fc_param_->param, fc_param_->algo_info,
                                                                          options.device->GetAllocator());

            if (bias_data != nullptr) {
                fc_param_->mgr->gen_cvt_weights(weight_data, bias_data);
            } else {
                std::vector<float> zero_bias(weight_shape.dims[0], 0.0f);
                fc_param_->mgr->gen_cvt_weights(weight_data, zero_bias.data());
            }
        }
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeGemm(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

bool GemmOp::TryFuseReLU() {
    gemm_fuse_relu_ = true;
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::x86::fc_fp32_algo::UNKNOWN) {
        ppl::kernel::x86::fc_fp32_param param = fc_param_->mgr->param();
        param.fuse_flag |= ppl::kernel::x86::fc_fuse_flag::RELU;
        fc_param_->mgr->set_param(param);
    }
    return true;
}

KernelImpl* GemmOp::CreateKernelImpl() const {
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::x86::fc_fp32_algo::UNKNOWN) {
        return CreateKernelImplWithParam<FCKernel>(fc_param_);
    } else {
        auto kernel = CreateKernelImplWithParam<GemmKernel>(param_.get());
        kernel->SetFuseReLU(gemm_fuse_relu_);
        return kernel;
    }
}

}}} // namespace ppl::nn::x86
