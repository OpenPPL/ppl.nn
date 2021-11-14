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

#include "ppl/nn/runtime/scheduler_common.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

// puts inputs/extra_inputs/outputs/constants into a vector
vector<EdgeObject*> InitObjectInUse(const ir::GraphTopo* topo, RuntimeGraph* graph) {
    vector<EdgeObject*> objects_in_use(topo->GetMaxEdgeId(), nullptr);

    for (auto it = graph->inputs.begin(); it != graph->inputs.end(); ++it) {
        auto eid = (*it)->GetEdge()->GetId();
        objects_in_use[eid] = *it;
    }
    for (auto it = graph->extra_inputs.begin(); it != graph->extra_inputs.end(); ++it) {
        auto eid = (*it)->GetEdge()->GetId();
        objects_in_use[eid] = *it;
    }
    for (auto it = graph->outputs.begin(); it != graph->outputs.end(); ++it) {
        auto eid = (*it)->GetEdge()->GetId();
        objects_in_use[eid] = *it;
    }
    for (auto it = graph->constants.begin(); it != graph->constants.end(); ++it) {
        auto eid = (*it)->GetEdge()->GetId();
        objects_in_use[eid] = *it;
    }

    return objects_in_use;
}

static RetCode AfterExecuteKernel(KernelImpl* kernel, KernelExecContext* ctx,
                                  const function<RetCode(EdgeObject*, nodeid_t)>& release_func) {
    auto nid = kernel->GetNode()->GetId();

    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto object = ctx->GetInput<EdgeObject>(i);
        if (!object) {
            continue;
        }

        auto status = release_func(object, nid);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "release edge[" << object->GetEdge()->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    for (uint32_t i = 0; i < ctx->GetExtraInputCount(); ++i) {
        auto object = ctx->GetExtraInput<EdgeObject>(i);
        if (!object) {
            continue;
        }

        auto status = release_func(object, nid);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "release edge[" << object->GetEdge()->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto object = ctx->GetOutput<EdgeObject>(i);
        auto status = release_func(object, nid);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "release edge[" << object->GetEdge()->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

RetCode ExecuteKernel(KernelImpl* kernel, KernelExecContext* ctx,
                      const function<RetCode(EdgeObject*, nodeid_t)>& release_func, Profiler* profiler) {
    auto exec_status = kernel->Execute(ctx);

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    profiler->CollectStatistics(kernel);
#endif

    auto status = AfterExecuteKernel(kernel, ctx, release_func);

    if (exec_status != RC_SUCCESS) {
        LOG(ERROR) << "exec kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(exec_status);
        return exec_status;
    }

    return status;
}

}}} // namespace ppl::nn::utils
