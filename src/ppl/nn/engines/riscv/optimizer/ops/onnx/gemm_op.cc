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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/engines/riscv/kernels/onnx/fc_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

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
    if (RISCV_USE_FP16 == options.engine_options->forward_precision) {
        return InitFC<__fp16>(options);
    } else if (RISCV_USE_FP32 == options.engine_options->forward_precision) {
        return InitFC<float>(options);
    } else {
        return RC_UNSUPPORTED;
    }
}

RetCode GemmOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (DATAFORMAT_N8CX == selected_input_formats->at(0)) {
        selected_output_formats->at(0) = DATAFORMAT_N8CX;
    } else if (DATAFORMAT_N4CX == selected_input_formats->at(0)) {
        selected_output_formats->at(0) = DATAFORMAT_N4CX;
    }

    return RC_SUCCESS;
}

RetCode GemmOp::SelectDataType(const InputOutputInfo& info, std::vector<datatype_t>* selected_input_data_types,
                               std::vector<datatype_t>* selected_output_data_types) {
    if (DATATYPE_FLOAT16 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    } else if (DATATYPE_FLOAT32 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT32;
    }

    return RC_SUCCESS;
}

bool GemmOp::TryFuseReLU() {
    gemm_fuse_relu_ = true;
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::riscv::fc_common_algo::unknown) {
        ppl::kernel::riscv::fc_common_param param = fc_param_->mgr->param();
        param.fuse_flag |= ppl::kernel::riscv::fc_fuse_flag::relu;
        fc_param_->mgr->set_param(param);
    }
    return true;
}

KernelImpl* GemmOp::CreateKernelImpl() const {
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::riscv::fc_common_algo::unknown) {
        return CreateKernelImplWithParam<FCKernel>(fc_param_);
    } else {
        auto kernel = CreateKernelImplWithParam<GemmKernel>(param_.get());
        kernel->SetFuseReLU(gemm_fuse_relu_);
        return kernel;
    }
}

}}} // namespace ppl::nn::riscv