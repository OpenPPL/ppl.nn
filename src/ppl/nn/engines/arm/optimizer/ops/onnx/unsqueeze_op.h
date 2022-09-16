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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_UNSQUEEZE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_UNSQUEEZE_OP_H_

#include "ppl/nn/params/onnx/unsqueeze_param.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/unsqueeze.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace arm {

class UnsqueezeOp final : public ArmOptKernel {
public:
    UnsqueezeOp(const ir::Node* node);
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    KernelImpl* CreateKernelImpl() const override;

#ifdef PPLNN_ENABLE_PMX_MODEL
    virtual ppl::nn::pmx::onnx::OpParamType GetOptParamType(void) const override {
        return ppl::nn::pmx::onnx::OpParamType_UnsqueezeParam;
    }

    virtual flatbuffers::Offset<void> SerializeOptParam(flatbuffers::FlatBufferBuilder* builder) const override {
        return ppl::nn::pmx::onnx::SerializeUnsqueezeParam(*param_.get(), builder).Union();
    }

    virtual ppl::common::RetCode DeserializeOptParam(const ppl::nn::pmx::onnx::OpParam* op_param) override {
        param_ = std::make_shared<ppl::nn::onnx::UnsqueezeParam>();
        ppl::nn::pmx::onnx::DeserializeUnsqueezeParam(*op_param->value_as_UnsqueezeParam(), param_.get());
        return ppl::common::RC_SUCCESS;
    }
#endif

private:
    std::shared_ptr<ppl::nn::onnx::UnsqueezeParam> param_;
};

}}} // namespace ppl::nn::arm

#endif
