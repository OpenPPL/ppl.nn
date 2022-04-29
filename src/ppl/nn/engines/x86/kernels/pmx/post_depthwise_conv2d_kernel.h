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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_PMX_POST_DEPTHWISE_CONV2D_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_PMX_POST_DEPTHWISE_CONV2D_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/engines/x86/params/post_depthwise_conv_param.h"
#include "ppl/kernel/x86/fp32/pd_conv2d.h"

namespace ppl { namespace nn { namespace x86 {

class PostDepthwiseConv2dKernel : public X86Kernel {
public:
    PostDepthwiseConv2dKernel(const ir::Node* node) : X86Kernel(node) {}
    ~PostDepthwiseConv2dKernel() {
        if (executor_) {
            delete executor_->conv2d_executor();
            delete executor_->depthwise_conv2d_executor();
            delete executor_;
        }
    }

    void SetParam(const PostDepthwiseConv2dParam* p) {
        param_ = p;
        if (executor_) {
            delete executor_->conv2d_executor();
            delete executor_->depthwise_conv2d_executor();
            delete executor_;
        }
        executor_ = p->mgr->gen_executor();
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    ppl::common::RetCode SeparateExecute(KernelExecContext* ctx, TensorImpl* X, TensorImpl* Y);
    ppl::common::RetCode FuseExecute(KernelExecContext* ctx, TensorImpl* X, TensorImpl* Y);

private:
    const PostDepthwiseConv2dParam* param_ = nullptr;
    ppl::kernel::x86::pd_conv2d_fp32_executor* executor_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
