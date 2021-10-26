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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_RULES_UTILS_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_RULES_UTILS_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/common/logger.h"
#include "ppl/common/retcode.h"

namespace ppl { namespace nn { namespace x86 {

inline ppl::common::RetCode CreateX86OptKernel(
    const OptKernelOptions& options,
    const ir::Node* node,
    X86OptKernel** kernel) {
    auto& type = node->GetType();

    auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for X86OptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                   << type.name << "]";
        return ppl::common::RC_NOT_FOUND;
    }

    auto opt_kernel = std::unique_ptr<X86OptKernel>(creator(node));
    if (!opt_kernel) {
        LOG(ERROR) << "create X86OptKernel failed: oom";
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << node->GetName() << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    *kernel = opt_kernel.get();
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));

    return ppl::common::RC_SUCCESS;
}

inline bool IsGraphOutput(const ir::GraphTopo* graph_topo, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph_topo->GetOutputCount(); i++) {
        if (graph_topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

}}} // namespace ppl::nn::x86

#endif
