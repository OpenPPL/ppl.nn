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

SequentialScheduler::SequentialScheduler() {
    acquire_object_func_ = [this](edgeid_t eid, uint32_t etype) -> EdgeObject* {
        if (eid >= edgeid2object_->size()) {
            return nullptr;
        }

        auto object = edgeid2object_->at(eid);
        if (!object) {
            auto edge = topo_->GetEdge(eid);

            if (etype == EdgeObject::T_TENSOR) {
                auto tensor = tensor_pool_.Alloc(edge, TENSORTYPE_NORMAL);
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
            edgeid2object_->at(eid) = object;
        }
        return object;
    };

    release_object_func_ = [this](EdgeObject* object, nodeid_t user) -> RetCode {
        auto eid = object->GetEdge()->GetId();
        if (edge_last_consumer_->at(eid) == user) {
            auto obj = edgeid2object_->at(eid);

            auto barrier = obj->GetBarrier();
            if (barrier) {
                auto status = barrier->Sync();
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "sync edge[" << obj->GetEdge()->GetName() << "] failed: " << GetRetCodeStr(status);
                    return status;
                }
            }

            if (obj->GetObjectType() == EdgeObject::T_TENSOR) {
                tensor_pool_.Free(static_cast<TensorImpl*>(obj));
            } else if (obj->GetObjectType() == EdgeObject::T_TENSOR_SEQUENCE) {
                tensor_sequence_pool_.Free(static_cast<TensorSequence*>(obj));
            } else {
                LOG(ERROR) << "invalid edge object type[" << obj->GetObjectType() << "]";
                return RC_INVALID_VALUE;
            }
            edgeid2object_->at(eid) = nullptr;
        }
        return RC_SUCCESS;
    };
}

RetCode SequentialScheduler::Init(const Options& options) {
    topo_ = options.topo;
    sorted_nodes_ = options.sorted_nodes;
    edge_last_consumer_ = options.edge_last_consumer;
    edgeid2object_ = options.edgeid2object;
    nodeid2kernel_ = options.nodeid2kernel;
    return RC_SUCCESS;
}

RetCode SequentialScheduler::DoForEach(const function<RetCode(KernelImpl*, KernelExecContext*)>& exec,
                                       Profiler* profiler) {
#ifndef NDEBUG
    set<edgeid_t> edges_before;
    for (uint32_t i = 0; i < edgeid2object_->size(); ++i) {
        if (edgeid2object_->at(i)) {
            edges_before.insert(i);
        }
    }
#endif

    KernelExecContext ctx;
    ctx.SetAcquireFunc(acquire_object_func_);
    ctx.SetProfilingFlag((profiler != nullptr));
    ctx.SetEdgeLastConsumerList(edge_last_consumer_);

    for (auto x = sorted_nodes_->begin(); x != sorted_nodes_->end(); ++x) {
        auto kernel = nodeid2kernel_->at(*x).get();
        ctx.SetNode(kernel->GetNode());

        auto exec_status = exec(kernel, &ctx);

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        if (profiler) {
            profiler->CollectStatistics(kernel);
        }
#endif

        auto status = utils::ReleaseKernelInputOutput(kernel, &ctx, release_object_func_);

        if (exec_status != RC_SUCCESS) {
            auto& type = kernel->GetNode()->GetType();
            LOG(ERROR) << "exec kernel[" << kernel->GetName() << "] of type[" << type.domain << ":" << type.name << ":"
                       << type.version << "] failed: " << GetRetCodeStr(exec_status);
            return exec_status;
        }

        if (status != RC_SUCCESS) {
            LOG(ERROR) << "release resources of kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

#ifndef NDEBUG
    set<edgeid_t> edges_after;
    for (uint32_t i = 0; i < edgeid2object_->size(); ++i) {
        if (edgeid2object_->at(i)) {
            edges_after.insert(i);
        }
    }

    vector<edgeid_t> diff_before2after(edges_before.size());
    auto end_iter = std::set_difference(edges_before.begin(), edges_before.end(), edges_after.begin(),
                                        edges_after.end(), diff_before2after.begin());
    diff_before2after.resize(end_iter - diff_before2after.begin());
    if (!diff_before2after.empty()) {
        LOG(ERROR) << "edge(s) in `before` but not in `after`:";
        for (auto x = diff_before2after.begin(); x != diff_before2after.end(); ++x) {
            auto edge = topo_->GetEdge(*x);
            LOG(ERROR) << " " << edge->GetName();
        }
    }

    vector<edgeid_t> diff_after2before(edges_after.size());
    end_iter = std::set_difference(edges_after.begin(), edges_after.end(), edges_before.begin(), edges_before.end(),
                                   diff_after2before.begin());
    diff_after2before.resize(end_iter - diff_after2before.begin());
    if (!diff_after2before.empty()) {
        LOG(ERROR) << "edge(s) in `after` but not in `before`:";
        for (auto x = diff_after2before.begin(); x != diff_after2before.end(); ++x) {
            auto edge = topo_->GetEdge(*x);
            LOG(ERROR) << " " << edge->GetName();
        }
    }

    if (!diff_before2after.empty() || !diff_after2before.empty()) {
        return RC_OTHER_ERROR;
    }
#endif

    return RC_SUCCESS;
}

RetCode SequentialScheduler::Run(Profiler* profiler) {
    return DoForEach(
        [](KernelImpl* kernel, KernelExecContext* ctx) -> RetCode {
            return kernel->Execute(ctx);
        },
        profiler);
}

RetCode SequentialScheduler::ForEach(const std::function<ppl::common::RetCode(KernelImpl*, KernelExecContext*)>& f) {
    return DoForEach(f, nullptr);
}

}} // namespace ppl::nn
