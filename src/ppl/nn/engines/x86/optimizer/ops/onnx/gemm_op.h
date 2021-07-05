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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_GEMM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_GEMM_OP_H_

#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/nn/engines/x86/params/fc_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class GemmOp final : public X86OptKernel {
public:
    GemmOp(const ir::Node* node) : X86OptKernel(node), fc_param_(nullptr) {}
    ~GemmOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    bool SetFuseReLU();

private:
    FCParam* fc_param_;
    std::shared_ptr<ppl::nn::common::GemmParam> param_;
    bool gemm_fuse_relu_ = false;
};

}}} // namespace ppl::nn::x86

#endif
