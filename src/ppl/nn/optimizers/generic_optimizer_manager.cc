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

#include "ppl/nn/optimizers/generic_optimizer_manager.h"
#include "ppl/nn/optimizers/constant_node_optimizer.h"
#include "ppl/nn/optimizers/identity_node_optimizer.h"
#include "ppl/nn/optimizers/skip_dropout_optimizer.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

GenericOptimizerManager::GenericOptimizerManager() {
    optimizer_list_.push_back(make_shared<ConstantNodeOptimizer>());
    optimizer_list_.push_back(make_shared<IdentityNodeOptimizer>());
    optimizer_list_.push_back(make_shared<SkipDropoutOptimizer>());
}

void GenericOptimizerManager::AppendOptimizer(const shared_ptr<GraphOptimizer>& opt) {
    optimizer_list_.push_back(opt);
}

RetCode GenericOptimizerManager::Process(ir::Graph* graph) const {
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
