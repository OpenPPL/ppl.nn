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

#ifndef _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_KERNEL_H_
#define _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_KERNEL_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/kernel_impl.h"

namespace ppl { namespace nn { namespace demo {

class DemoKernel final : public KernelImpl {
public:
    DemoKernel(const ir::Node* node) : KernelImpl(node) {}
    DemoKernel(DemoKernel&&) = default;

    ppl::common::RetCode Execute(KernelExecContext* ctx) override;
};

class DemoOptKernel final : public OptKernel {
public:
    DemoOptKernel(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new DemoKernel(GetNode());
    }
};

}}} // namespace ppl::nn::demo

#endif
