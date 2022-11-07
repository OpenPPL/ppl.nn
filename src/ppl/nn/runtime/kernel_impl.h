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

#ifndef _ST_HPC_PPL_NN_RUNTIME_KERNEL_IMPL_H_
#define _ST_HPC_PPL_NN_RUNTIME_KERNEL_IMPL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/kernel_exec_context.h"
#include "ppl/nn/engines/engine_context.h"
#include <string>

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include "ppl/nn/runtime/internal_profiling_info.h"
#endif

namespace ppl { namespace nn {

class KernelImpl {
public:
    KernelImpl(const ir::Node* node) : node_(node), eng_ctx_(nullptr) {}
    KernelImpl(KernelImpl&&) = default;
    KernelImpl& operator=(KernelImpl&&) = default;
    virtual ~KernelImpl() {}

    /** @brief get associated node in the compute graph */
    const ir::Node* GetNode() const {
        return node_;
    }

    /** @brief get kernel's name */
    const std::string& GetName() const {
        return node_->GetName();
    }

    /** @brief get kernel's type */
    const ir::Node::Type& GetType() const {
        return node_->GetType();
    }

    /** @brief set engine context that this kernel uses during execution. */
    void SetEngineContext(EngineContext* ctx) {
        eng_ctx_ = ctx;
    }

    /** @brief get the `EngineContext` which this kernel uses. */
    EngineContext* GetEngineContext() const {
        return eng_ctx_;
    }

    /** @brief infer output shapes '*/
    virtual ppl::common::RetCode Reshape(InputOutputInfo*) const = 0;

    /**
       @brief evaluate this op.
       @param ctx contexts needed during execution
    */
    virtual ppl::common::RetCode Execute(KernelExecContext* ctx) = 0;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
public:
    virtual void GetProfilingInfo(InternalProfilingInfo*) const {}
#endif

private:
    /** assiciated node in the compute graph */
    const ir::Node* node_;

    /** EngineContext which this kernel uses */
    EngineContext* eng_ctx_;

private:
    KernelImpl(const KernelImpl&) = delete;
    KernelImpl& operator=(const KernelImpl&) = delete;
};

}} // namespace ppl::nn

#endif
