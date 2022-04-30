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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/conv/conv_op.h"
#include "ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/fp16/conv2d.h"
#include "ppl/nn/engines/riscv/kernels/onnx/conv/conv2d_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_conv.h"
#include "ppl/nn/engines/riscv/riscv_engine_options.h"
#include "ppl/nn/common/logger.h"
#include <cstring>

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

ConvOp::~ConvOp() {
    if (conv2d_param_ != nullptr) {
        if (conv2d_param_->mgr != nullptr) {
            conv2d_param_->mgr->release_cvt_weights();
            delete conv2d_param_->mgr;
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
    if (DATATYPE_FLOAT16 == options.engine_options->forward_precision) {
        return SelectAlgorithmGeneric<__fp16>(info, options);
    } else if (DATATYPE_FLOAT32 == options.engine_options->forward_precision) {
        return SelectAlgorithmGeneric<float>(info, options);
    } else {
        return RC_UNSUPPORTED;
    }
}

RetCode ConvOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (conv2d_param_ && conv2d_param_->mgr &&
        conv2d_param_->mgr->algo_info().algo_type != ppl::kernel::riscv::conv2d_common_algo::unknown) {
        selected_input_formats->at(0) = conv2d_param_->mgr->algo_info().input_format;
        selected_output_formats->at(0) = conv2d_param_->mgr->algo_info().output_format;
        return RC_SUCCESS;
    }
    return RC_INVALID_VALUE;
}

ppl::common::RetCode ConvOp::SelectDataType(const InputOutputInfo& info, ppl::common::datatype_t forward_precision,
                                            std::vector<ppl::common::datatype_t>* selected_input_data_types,
                                            std::vector<ppl::common::datatype_t>* selected_output_data_types) {
    if (conv2d_param_ && conv2d_param_->mgr &&
        conv2d_param_->mgr->algo_info().algo_type != ppl::kernel::riscv::conv2d_common_algo::unknown) {
        selected_input_data_types->at(0) = conv2d_param_->mgr->algo_info().input_data_type;
        selected_output_data_types->at(0) = conv2d_param_->mgr->algo_info().output_data_type;
        return RC_SUCCESS;
    }
    return RC_INVALID_VALUE;
}

bool ConvOp::TryFuseReLU(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::riscv::conv2d_common_algo::unknown) {
        return false;
    }
    ppl::kernel::riscv::conv2d_common_param param = conv2d_param_->mgr->param();
    param.fuse_flag |= ppl::kernel::riscv::conv_fuse_flag::RELU;
    conv2d_param_->mgr->set_param(param);
    return true;
}

bool ConvOp::TryFuseReLU6(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::riscv::conv2d_common_algo::unknown) {
        return false;
    }
    ppl::kernel::riscv::conv2d_common_param param = conv2d_param_->mgr->param();
    param.fuse_flag |= ppl::kernel::riscv::conv_fuse_flag::RELU;
    param.fuse_flag |= ppl::kernel::riscv::conv_fuse_flag::RELU6;
    conv2d_param_->mgr->set_param(param);
    return true;
}

bool ConvOp::TryFuseSum(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::riscv::conv2d_common_algo::unknown) {
        return false;
    }
    ppl::kernel::riscv::conv2d_common_param param = conv2d_param_->mgr->param();
    if (param.fuse_flag) { // already fused sum, relu or relu6
        return false;
    }
    param.fuse_flag |= ppl::kernel::riscv::conv_fuse_flag::SUM;
    conv2d_param_->mgr->set_param(param);
    return true;
}

KernelImpl* ConvOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<Conv2dKernel>(conv2d_param_);
}

}}} // namespace ppl::nn::riscv
