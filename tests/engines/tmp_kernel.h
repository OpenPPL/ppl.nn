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

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/opt_kernel.h"

namespace ppl { namespace nn { namespace test {

class TmpKernelOne final : public KernelImpl {
public:
    TmpKernelOne(const ir::Node* node) : KernelImpl(node) {}
    ppl::common::RetCode Execute(KernelExecContext*) override {
        return ppl::common::RC_SUCCESS;
    };
};

class TmpKernelTwo final : public KernelImpl {
public:
    TmpKernelTwo(const ir::Node* node) : KernelImpl(node) {}
    ppl::common::RetCode Execute(KernelExecContext*) override {
        return ppl::common::RC_SUCCESS;
    };
};

class TmpOptKernelOne : public OptKernel {
public:
    TmpOptKernelOne(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new TmpKernelOne(GetNode());
    }
};

class TmpOptKernelTwo : public OptKernel {
public:
    TmpOptKernelTwo(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new TmpKernelTwo(GetNode());
    }
};

}}} // namespace ppl::nn::test
