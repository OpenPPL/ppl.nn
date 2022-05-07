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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_RESHAPE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPS_ONNX_RESHAPE_OP_H_

#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace arm {

class ReshapeOp final : public ArmOptKernel {
public:
    ReshapeOp(const ir::Node* node);
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    ppl::common::RetCode SelectAlgorithm(const InputOutputInfo&, const OptKernelOptions&);
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_types,
                                        const ppl::common::datatype_t preferred_fp_datatype) override;
    KernelImpl* CreateKernelImpl() const override;
};

}}} // namespace ppl::nn::arm

#endif
