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

    const_object_refcount_ = utils::InitObjectRefcount(topo_);
    edgeid2object_ = utils::InitObjectInUse(topo_, g);

    return RC_SUCCESS;
}

RetCode SequentialScheduler::Run(Profiler* profiler) {
    std::vector<uint32_t> object_refcount = const_object_refcount_;

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

    auto release_object_func = [this, &object_refcount](EdgeObject* object) -> RetCode {
        auto eid = object->GetEdge()->GetId();
        uint32_t& refcount = object_refcount[eid];
        if (refcount > 0) {
            --refcount;
            if (refcount == 0 && edgeid2object_[eid]) {
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
        }

        LOG(ERROR) << "invalid refcount of object[" << object->GetEdge()->GetName() << "]";
        return RC_INVALID_VALUE;
    };

    auto get_barrier_func = [this](edgeid_t eid) -> Barrier* {
        if (eid >= edgeid2object_.size()) {
            return nullptr;
        }

        return graph_->edgeid2barrier[eid].get();
    };

    KernelExecContext ctx;
    ctx.SetAcquireObjectFunc(acquire_object_func);
    ctx.SetGetBarrierFunc(get_barrier_func);
    ctx.SetProfilingFlag(profiler->IsProfilingEnabled());

    for (auto x = aux_info_->sorted_nodes.begin(); x != aux_info_->sorted_nodes.end(); ++x) {
        auto kernel = graph_->nodeid2kernel[*x].get();
        ctx.SetNode(kernel->GetNode());
        ctx.SetDevice(kernel->GetDevice());

        auto status =
            utils::ExecuteKernel(kernel, &ctx, graph_->kernel_barrier_flag[*x], release_object_func, profiler);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "execute kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
