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

#include "ppl/nn/optimizers/graph_optimizer_manager.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/optimizers/constant_node_optimizer.h"
#include "ppl/nn/optimizers/fuse_parallel_node_optimizer.h"
#include "ppl/nn/optimizers/fuse_bn_optimizer.h"
#include "ppl/nn/optimizers/fuse_shape_optimizer.h"
#include "ppl/nn/optimizers/skip_dropout_optimizer.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

#define REGISTER_OPTIMIZER(name, type) name2optimizer_.emplace(name, unique_ptr<GraphOptimizer>(new type()))

GraphOptimizerManager::GraphOptimizerManager() {
    REGISTER_OPTIMIZER("ConstantNodeOptimizer", ConstantNodeOptimizer);
    REGISTER_OPTIMIZER("FuseParallelNodeOptimizer", FuseParallelNodeOptimizer);
    REGISTER_OPTIMIZER("FuseBNOptimizer", FuseBNOptimizer);
    REGISTER_OPTIMIZER("FuseShapeOptimizer", FuseShapeOptimizer);
    REGISTER_OPTIMIZER("SkipDropoutOptimizer", SkipDropoutOptimizer);
}

RetCode GraphOptimizerManager::Process(ir::Graph* graph) const {
    for (auto x = name2optimizer_.begin(); x != name2optimizer_.end(); ++x) {
        auto status = x->second->Optimize(graph);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "optimizer[" << x->first << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
