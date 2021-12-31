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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_KERNELS_ONNX_MAXPOOL_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_KERNELS_ONNX_MAXPOOL_KERNEL_H_

#include "ppl/nn/engines/riscv/kernel.h"
#include "ppl/nn/params/onnx/pooling_param.h"

namespace ppl { namespace nn { namespace riscv {

class MaxPoolKernel : public RISCVKernel {
public:
    MaxPoolKernel(const ir::Node* node) : RISCVKernel(node) {}

    void SetParam(const ppl::nn::common::PoolingParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::PoolingParam* param_ = nullptr;
};

}}} // namespace ppl::nn::riscv

#endif