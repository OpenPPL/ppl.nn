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

private:
    ppl::nn::arm::FCParam* fc_param_;
    std::shared_ptr<ppl::nn::onnx::GemmParam> param_;
    bool gemm_fuse_relu_ = false;
};

}}} // namespace ppl::nn::arm

#endif
