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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNEL_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/engines/x86/macros.h"
#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/engines/x86/x86_common_param.h"
#include "ppl/common/sys.h"

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include <chrono>
#endif

namespace ppl { namespace nn { namespace x86 {

class X86Kernel : public KernelImpl {
public:
    X86Kernel(const ir::Node* node) : KernelImpl(node), common_param_(nullptr) {}
    virtual ~X86Kernel() {}

    ppl::common::RetCode Execute(KernelExecContext*) override final;

    ppl::common::RetCode Reshape(KernelExecContext* ctx) const {
        return reshape_func_(ctx);
    }

    void SetReshapeFunc(const std::function<ppl::common::RetCode(InputOutputInfo*)>& f) {
        reshape_func_ = f;
    }

    void SetCommonParam(const X86CommonParam* p) {
        common_param_ = p;
    }

protected:
    virtual bool CanDoExecute(const KernelExecContext&) const;

    virtual ppl::common::RetCode DoExecute(KernelExecContext*) = 0;
    virtual uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const {
        return 0;
    }

    bool MayUseISA(uint32_t flag) const {
        return !!(GetX86Device()->GetISA() & flag);
    }
    uint32_t GetISA() const {
        return GetX86Device()->GetISA();
    }

    X86Device* GetX86Device() {
        return reinterpret_cast<X86Device*>(GetDevice());
    }
    const X86Device* GetX86Device() const {
        return reinterpret_cast<const X86Device*>(GetDevice());
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
public:
    uint64_t GetExecutionTime() const override {
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end_ts_ - begin_ts_);
        return diff.count();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> begin_ts_;
    std::chrono::time_point<std::chrono::system_clock> end_ts_;
#endif

private:
    ppl::common::RetCode BeforeExecute(KernelExecContext*);

private:
    const X86CommonParam* common_param_ = nullptr;
    std::function<ppl::common::RetCode(InputOutputInfo*)> reshape_func_;
};

}}} // namespace ppl::nn::x86

#endif
