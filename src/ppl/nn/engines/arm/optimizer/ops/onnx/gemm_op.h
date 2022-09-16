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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_GEMM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_GEMM_OP_H_

#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/nn/engines/arm/params/fc_param.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/gemm.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace arm {

class GemmOp final : public ArmOptKernel {
public:
    GemmOp(const ir::Node* node);
    ~GemmOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    ppl::common::RetCode SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_types,
                                        const ppl::common::datatype_t preferred_fp_datatype) override;
    KernelImpl* CreateKernelImpl() const override;
    bool TryFuseReLU();

#ifdef PPLNN_ENABLE_PMX_MODEL
    virtual ppl::nn::pmx::onnx::OpParamType GetOptParamType(void) const override {
        return ppl::nn::pmx::onnx::OpParamType_GemmParam;
    }

    virtual flatbuffers::Offset<void> SerializeOptParam(flatbuffers::FlatBufferBuilder* builder) const override {
        return ppl::nn::pmx::onnx::SerializeGemmParam(*param_.get(), builder).Union();
    }

    virtual ppl::common::RetCode DeserializeOptParam(const ppl::nn::pmx::onnx::OpParam* op_param) override {
        param_ = std::make_shared<ppl::nn::onnx::GemmParam>();
        ppl::nn::pmx::onnx::DeserializeGemmParam(*op_param->value_as_GemmParam(), param_.get());
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode SerializeData(const ::ppl::nn::pmx::SerializationContext&, utils::DataStream*) const override;
    ppl::common::RetCode DeserializeData(const ::ppl::nn::pmx::DeserializationContext&, const void*, uint64_t) override;
#endif

    virtual void SetAllocator(ppl::common::Allocator *allocator) override {
        this->allocator_ = allocator;
    }

private:
    ppl::nn::arm::FCParam* fc_param_;
    std::shared_ptr<ppl::nn::onnx::GemmParam> param_;
    bool gemm_fuse_relu_ = false;

    ppl::common::Allocator *allocator_;
};

}}} // namespace ppl::nn::arm

#endif
