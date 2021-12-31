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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPS_ONNX_CAST_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPS_ONNX_CAST_OP_H_

#include "ppl/nn/params/onnx/cast_param.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace riscv {

class CastOp final : public RISCVOptKernel {
public:
    CastOp(const ir::Node* node) : RISCVOptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;

private:
    std::shared_ptr<ppl::nn::common::CastParam> param_;
};

}}} // namespace ppl::nn::riscv

#endif
