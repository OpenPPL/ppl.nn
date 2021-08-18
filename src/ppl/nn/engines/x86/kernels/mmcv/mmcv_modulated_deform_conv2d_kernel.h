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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_MMCV_MODULATED_DEFORM_CONV2D_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_MMCV_MODULATED_DEFORM_CONV2D_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/mmcv/mmcv_modulated_deform_conv2d_param.h"

namespace ppl { namespace nn { namespace x86 {

class MMCVModulatedDeformConv2dKernel : public X86Kernel {
public:
    MMCVModulatedDeformConv2dKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::MMCVModulatedDeformConv2dParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::MMCVModulatedDeformConv2dParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
