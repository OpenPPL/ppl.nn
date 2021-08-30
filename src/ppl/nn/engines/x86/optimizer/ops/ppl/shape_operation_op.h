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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_PPL_SHAPE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_PPL_SHAPE_OP_H_

#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

#include "ppl/nn/engines/common/ppl/shape_operation_op.h"

namespace ppl { namespace nn { namespace x86 {

class PPLShapeOperationOp final : public X86OptKernel {
public:
    PPLShapeOperationOp(const ir::Node* node) : X86OptKernel(node), op_(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;

private:
    std::shared_ptr<ppl::nn::common::PPLShapeOperationParam> param_;
    ppl::nn::common::PPLShapeOperationOp op_;
};

}}} // namespace ppl::nn::x86

#endif
