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

#include "ppl/nn/optimizers/nn_optimizer_manager.h"
#include "ppl/nn/optimizers/fuse_parallel_node_optimizer.h"
#include "ppl/nn/optimizers/fuse_bn_optimizer.h"
#include "ppl/nn/optimizers/fuse_constant_optimizer.h"
#include "ppl/nn/optimizers/fuse_shape_optimizer.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

NNOptimizerManager::NNOptimizerManager() {
    optimizer_list_.push_back(make_shared<FuseParallelNodeOptimizer>());
    optimizer_list_.push_back(make_shared<FuseBNOptimizer>());
    optimizer_list_.push_back(make_shared<FuseConstantOptimizer>());
    optimizer_list_.push_back(make_shared<FuseShapeOptimizer>());
}

void NNOptimizerManager::AppendOptimizer(const shared_ptr<GraphOptimizer>& opt) {
    optimizer_list_.push_back(opt);
}

RetCode NNOptimizerManager::Process(ir::Graph* graph) const {
    for (auto x = optimizer_list_.begin(); x != optimizer_list_.end(); ++x) {
        auto status = (*x)->Optimize(graph);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "optimizer[" << (*x)->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
