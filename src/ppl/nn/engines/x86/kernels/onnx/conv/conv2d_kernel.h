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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_CONV_CONV2D_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_CONV_CONV2D_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/engines/x86/params/convolution_param.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace nn { namespace x86 {

class Conv2dKernel : public X86Kernel {
public:
    Conv2dKernel(const ir::Node* node) : X86Kernel(node) {}
    ~Conv2dKernel() {
        if (executor_)
            delete executor_;
    }

    void SetParam(const Convolution2DParam* p) {
        param_ = p;
        if (executor_)
            delete executor_;
        executor_ = p->mgr->gen_executor();
        if (p->fallback_mgr) {
            if (fallback_executor_)
                delete fallback_executor_;
            fallback_executor_ = p->fallback_mgr->gen_executor();
        }
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const Convolution2DParam* param_ = nullptr;
    ppl::kernel::x86::conv2d_fp32_executor* executor_ = nullptr;
    ppl::kernel::x86::conv2d_fp32_executor* fallback_executor_ = nullptr;
    bool use_fallback_ = false;
};

}}} // namespace ppl::nn::x86

#endif
