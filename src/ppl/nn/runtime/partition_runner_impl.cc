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

#include "ppl/nn/ir/partial_graph_topo.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/runtime/utils.h"
#include "ppl/nn/runtime/partition_runner_impl.h"
#include "ppl/nn/runtime/sequential_scheduler.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RetCode PartitionRunnerImpl::Init(const shared_ptr<ir::GraphTopo>& topo, const vector<nodeid_t>& nodes,
                                  vector<unique_ptr<EngineContext>>* engctx, vector<EdgeObject*>* e2o,
                                  vector<unique_ptr<KernelImpl>>* n2k) {
    topo_.reset(new ir::PartialGraphTopo(topo.get(), nodes));

    utils::DfsDeeperFirst(topo_.get(), [this](nodeid_t nid) -> void {
        sorted_nodes_.push_back(nid);
    });

    auto rc = utils::GenEdgeLastConsumer(topo_.get(), sorted_nodes_, {}, &edge_last_consumer_);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "GenEdgeLastConsumer failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // mark reserved tensors from parent scope
    for (size_t i = 0; i < e2o->size(); ++i) {
        if (e2o->at(i)) {
            edge_last_consumer_[i] = INVALID_NODEID;
        }
    }

    sched_ = unique_ptr<Scheduler>(new SequentialScheduler());
    rc = sched_->Init(Scheduler::Options(topo_.get(), &sorted_nodes_, &edge_last_consumer_, e2o, n2k));
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init scheduler failed: " << GetRetCodeStr(rc);
        return rc;
    }

    engctx_ = engctx;

    return RC_SUCCESS;
}

RetCode PartitionRunnerImpl::Sync() {
    for (auto e = engctx_->begin(); e != engctx_->end(); ++e) {
        auto dev = e->get()->GetDevice();
        if (dev) {
            auto rc = dev->Sync();
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "sync device[" << e->get()->GetName() << "] failed: " << GetRetCodeStr(rc);
                return rc;
            }
        }
    }
    return RC_SUCCESS;
}

RetCode PartitionRunnerImpl::Run() {
    auto rc = sched_->Run(nullptr);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "PartitionRunner Run() failed: " << GetRetCodeStr(rc);
        return rc;
    }

    return Sync();
}

}} // namespace ppl::nn
