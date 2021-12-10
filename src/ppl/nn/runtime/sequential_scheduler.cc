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

#ifndef NDEBUG
#include <set>
#endif

#include "ppl/nn/common/logger.h"
#include "ppl/nn/runtime/sequential_scheduler.h"
#include "ppl/nn/runtime/scheduler_common.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RetCode SequentialScheduler::Init(const ir::GraphTopo* topo, const RuntimeAuxInfo* aux_info, RuntimeGraph* g) {
    graph_ = g;
    topo_ = topo;
    aux_info_ = aux_info;
    edgeid2object_ = utils::InitObjectInUse(topo, g);
    return RC_SUCCESS;
}

#ifndef NDEBUG
static bool ValidateEdges(const RuntimeGraph* graph, const vector<EdgeObject*> eid2obj) {
    set<EdgeObject*> tensors_before;
    for (auto x = graph->inputs.begin(); x != graph->inputs.end(); ++x) {
        tensors_before.insert(*x);
    }
    for (auto x = graph->extra_inputs.begin(); x != graph->extra_inputs.end(); ++x) {
        tensors_before.insert(*x);
    }
    for (auto x = graph->constants.begin(); x != graph->constants.end(); ++x) {
        tensors_before.insert(*x);
    }
    for (auto x = graph->outputs.begin(); x != graph->outputs.end(); ++x) {
        tensors_before.insert(*x);
    }

    set<EdgeObject*> tensors_after;
    for (auto x = eid2obj.begin(); x != eid2obj.end(); ++x) {
        if (*x) {
            tensors_after.insert(*x);
        }
    }

    vector<EdgeObject*> diff_before2after(tensors_before.size());
    auto end_iter = std::set_difference(tensors_before.begin(), tensors_before.end(), tensors_after.begin(),
                                        tensors_after.end(), diff_before2after.begin());
    diff_before2after.resize(end_iter - diff_before2after.begin());
    if (!diff_before2after.empty()) {
        LOG(ERROR) << "edge(s) in `before` but not in `after`:";
        for (auto x = diff_before2after.begin(); x != diff_before2after.end(); ++x) {
            LOG(ERROR) << " " << (*x)->GetEdge()->GetName();
        }
    }

    vector<EdgeObject*> diff_after2before(tensors_after.size());
    end_iter = std::set_difference(tensors_after.begin(), tensors_after.end(), tensors_before.begin(),
                                   tensors_before.end(), diff_after2before.begin());
    diff_after2before.resize(end_iter - diff_after2before.begin());
    if (!diff_after2before.empty()) {
        LOG(ERROR) << "edge(s) in `after` but not in `before`:";
        for (auto x = diff_after2before.begin(); x != diff_after2before.end(); ++x) {
            LOG(ERROR) << " " << (*x)->GetEdge()->GetName();
        }
    }

    return (diff_before2after.empty() && diff_after2before.empty());
}
#endif

RetCode SequentialScheduler::Run(Profiler* profiler) {
    auto acquire_object_func = [this](edgeid_t eid, uint32_t etype, Device* device) -> EdgeObject* {
        if (eid >= edgeid2object_.size()) {
            return nullptr;
        }

        auto object = edgeid2object_[eid];
        if (!object) {
            auto edge = topo_->GetEdgeById(eid);

            if (etype == EdgeObject::T_TENSOR) {
                auto tensor = tensor_pool_.Alloc(edge, TENSORTYPE_NORMAL);
                tensor->SetDevice(device);
                object = tensor;
            } else if (etype == EdgeObject::T_TENSOR_SEQUENCE) {
                object = tensor_sequence_pool_.Alloc(edge);
            } else if (etype == EdgeObject::T_EDGE_OBJECT) {
                return nullptr;
            } else {
                LOG(ERROR) << "invalid object type[" << etype << "] of edge[" << edge->GetName() << "]";
                return nullptr;
            }

            if (!object) {
                LOG(ERROR) << "create output object[" << edge->GetName() << "] failed, oom";
                return nullptr;
            }
            edgeid2object_[eid] = object;
        }
        return object;
    };

    auto release_object_func = [this](EdgeObject* object, nodeid_t user) -> RetCode {
        auto eid = object->GetEdge()->GetId();
        if (aux_info_->tensor_last_consumer[eid] == user) {
            auto obj = edgeid2object_[eid];
            if (obj->GetObjectType() == EdgeObject::T_TENSOR) {
                tensor_pool_.Free(static_cast<TensorImpl*>(obj));
            } else if (obj->GetObjectType() == EdgeObject::T_TENSOR_SEQUENCE) {
                tensor_sequence_pool_.Free(static_cast<TensorSequence*>(obj));
            } else {
                LOG(ERROR) << "invalid edge object type[" << obj->GetObjectType() << "]";
                return RC_INVALID_VALUE;
            }
            edgeid2object_[eid] = nullptr;
        }
        return RC_SUCCESS;
    };

    KernelExecContext ctx;
    ctx.SetAcquireObjectFunc(acquire_object_func);
    ctx.SetProfilingFlag(profiler->IsProfilingEnabled());

    for (auto x = aux_info_->sorted_nodes.begin(); x != aux_info_->sorted_nodes.end(); ++x) {
        auto kernel = graph_->nodeid2kernel[*x].get();
        ctx.SetNode(kernel->GetNode());
        ctx.SetDevice(kernel->GetDevice());

        auto status = utils::ExecuteKernel(kernel, &ctx, release_object_func, profiler);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "execute kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

#ifndef NDEBUG
    if (!ValidateEdges(graph_, edgeid2object_)) {
        LOG(ERROR) << "valid edge(s) not matched.";
        return RC_INVALID_VALUE;
    }
#endif

    return RC_SUCCESS;
}

}} // namespace ppl::nn
