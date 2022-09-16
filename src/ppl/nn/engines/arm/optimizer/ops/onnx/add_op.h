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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_ADD_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_ADD_OP_H_

#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"
#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace arm {

class AddOp final : public ArmOptKernel {
public:
    AddOp(const ir::Node* node);
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    ppl::common::RetCode SelectAlgorithm(const InputOutputInfo&, const OptKernelOptions&) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    KernelImpl* CreateKernelImpl() const override;

    bool TryFuseReLU(void) {
        fuse_relu_ = true;
        return true;
    }
    bool HasFuseReLU(void) {
        return fuse_relu_;
    }

#ifdef PPLNN_ENABLE_PMX_MODEL
    virtual ppl::nn::pmx::arm::PrivateDataType GetPrivateDataType(void) const {
        return ppl::nn::pmx::arm::PrivateDataType_FusionData;
    }

    virtual flatbuffers::Offset<void> SerializePrivateData(flatbuffers::FlatBufferBuilder* builder) const {
        return ppl::nn::pmx::arm::CreateFusionData(*builder, (int8_t)(fuse_relu_ ? 1 : 0)).Union();
    }
    
    virtual ppl::common::RetCode DeserializePrivateData(const ppl::nn::pmx::arm::OpData* op_data) {
        fuse_relu_ = (op_data->value_as_FusionData()->fuse_relu() == (int8_t)1);
        return ppl::common::RC_SUCCESS;
    }
#endif

private:
    bool fuse_relu_ = false;
};

}}} // namespace ppl::nn::arm

#endif
