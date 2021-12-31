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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_KERNELS_ONNX_FC_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_KERNELS_ONNX_FC_KERNEL_H_

#include "ppl/nn/engines/riscv/kernel.h"
#include "ppl/nn/engines/riscv/params/fc_param.h"
#include "ppl/kernel/riscv/fp16/fc.h"

namespace ppl { namespace nn { namespace riscv {

class FCKernel : public RISCVKernel {
public:
    FCKernel(const ir::Node* node) : RISCVKernel(node) {}
    ~FCKernel() {
        if (executor_)
            delete executor_;
    }

    void SetParam(const FCParam* p) {
        if (executor_)
            delete executor_;
        executor_ = p->mgr->gen_executor();
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    ppl::kernel::riscv::fc_base_executor* executor_ = nullptr;
};

}}} // namespace ppl::nn::riscv

#endif