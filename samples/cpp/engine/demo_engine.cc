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

#include "demo_engine.h"
#include "demo_kernel.h"
#include "demo_engine_context.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace demo {

EngineContext* DemoEngine::CreateEngineContext() {
    return new DemoEngineContext();
}

static RetCode FillKernels(const ir::Graph* graph, RuntimePartitionInfo* info) {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        info->kernels.emplace(node->GetId(), unique_ptr<OptKernel>(new DemoOptKernel(node)));
    }
    return RC_SUCCESS;
}

RetCode DemoEngine::ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = utils::LoadConstants(*graph, &device_, &info->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "FillConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    status = FillKernels(graph, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "FillKernels failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::demo
