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
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/runtime/sequential_scheduler.h"
#include "ppl/nn/utils/utils.h"
#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RuntimeImpl::~RuntimeImpl() {
    sched_.reset();
    graph_.Clear();
    engctx_.clear();
    graph_info_.reset();
}

static EngineContext* FindOrCreateEngineContext(EngineImpl* engine, map<EngineImpl*, EngineContext*>* eng2ctx,
                                                vector<unique_ptr<EngineContext>>* engctx) {
    auto ref = eng2ctx->find(engine);
    if (ref != eng2ctx->end()) {
        return ref->second;
    }

    auto ctx = engine->CreateEngineContext();
    if (!ctx) {
        LOG(ERROR) << "create EngineContext for engine[" << engine->GetName() << "] failed.";
        return nullptr;
    }

    eng2ctx->insert(make_pair(engine, ctx));
    engctx->emplace_back(unique_ptr<EngineContext>(ctx));
    return ctx;
}

static RetCode InitRuntimeGraphResourceKernels(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                               const vector<bool>& valid_node_flags,
                                               vector<unique_ptr<EngineContext>>* engctx, RuntimeGraphResource* graph) {
    map<EngineImpl*, EngineContext*> eng2ctx;
    for (auto partition = info.partitions.begin(); partition != info.partitions.end(); ++partition) {
        auto ctx = FindOrCreateEngineContext(partition->engine, &eng2ctx, engctx);
        if (!ctx) {
            LOG(ERROR) << "create EngineContext for engine[" << partition->engine->GetName() << "] failed.";
            return RC_OTHER_ERROR;
        }

        for (auto o = partition->ops.begin(); o != partition->ops.end(); ++o) {
            if (!valid_node_flags[o->get()->GetNode()->GetId()]) {
                continue;
            }

            auto impl = (*o)->CreateKernelImpl();
            if (!impl) {
                LOG(ERROR) << "create kernel[" << (*o)->GetNode()->GetName() << "] failed.";
                return RC_OTHER_ERROR;
            }
            impl->SetDevice(ctx->GetDevice());
            graph->nodeid2kernel[(*o)->GetNode()->GetId()].reset(impl);
        }
    }

    return RC_SUCCESS;
}

static RetCode InitRuntimeGraphResourceInputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                              const map<string, nodeid_t>& name2nodeid, RuntimeGraphResource* graph) {
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        auto edge = topo->GetEdge(eid);
        auto ret_pair = graph->tensors.insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
        auto tensor = &ret_pair.first->second;

        if (ret_pair.second) {
            // finds a consumer to get device for this input
            for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer = topo->GetNode(it.Get());
                if (utils::IsPplConverterNode(consumer)) {
                    continue;
                }

                auto nid_ref = name2nodeid.find(consumer->GetName());
                if (nid_ref == name2nodeid.end()) {
                    LOG(ERROR) << "cannot find consumer[" << consumer->GetName() << "] of [" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }

                // Consumers of an input are in the same engine. This is guranteed by optimizer.
                auto kernel = graph->nodeid2kernel[nid_ref->second].get();
                tensor->SetDevice(kernel->GetDevice());
                break;
            }

            // ONNX supports reshaping inputs in runtime stage
            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }

        graph->edgeid2object[eid] = tensor;
    }

    return RC_SUCCESS;
}

static RetCode InitRuntimeGraphResourceExtraInputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                                   const map<string, nodeid_t>& name2nodeid,
                                                   RuntimeGraphResource* graph) {
    for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
        auto eid = topo->GetExtraInput(i);
        auto edge = topo->GetEdge(eid);
        auto ret_pair = graph->tensors.insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
        auto tensor = &ret_pair.first->second;

        if (ret_pair.second) {
            // finds a consumer to get device for this extra input
            for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer = topo->GetNode(it.Get());
                if (utils::IsPplConverterNode(consumer)) {
                    continue;
                }

                auto nid_ref = name2nodeid.find(consumer->GetName());
                if (nid_ref == name2nodeid.end()) {
                    LOG(ERROR) << "cannot find consumer[" << consumer->GetName() << "] of [" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }

                // Consumers of an input are in the same engine. This is guranteed by optimizer.
                auto kernel = graph->nodeid2kernel[nid_ref->second].get();
                tensor->SetDevice(kernel->GetDevice());
                break;
            }

            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }

        graph->edgeid2object[eid] = tensor;
    }

    return RC_SUCCESS;
}

RetCode InitRuntimeGraphResourceOutputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                        const map<string, nodeid_t>& name2nodeid, RuntimeGraphResource* graph) {
    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        auto eid = topo->GetOutput(i);
        auto edge = topo->GetEdge(eid);

        auto ret_pair = graph->tensors.insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
        auto tensor = &ret_pair.first->second;

        if (ret_pair.second) {
            auto producer_id = edge->GetProducer();
            if (producer_id != INVALID_NODEID) {
                auto producer = topo->GetNode(producer_id);

                auto nid_ref = name2nodeid.find(producer->GetName());
                if (nid_ref == name2nodeid.end()) {
                    LOG(ERROR) << "cannot find producer[" << producer->GetName() << "] of [" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }

                auto kernel = graph->nodeid2kernel[nid_ref->second].get();
                tensor->SetDevice(kernel->GetDevice());
            }

            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }

        graph->edgeid2object[eid] = tensor;
    }

    return RC_SUCCESS;
}

static RetCode InitRuntimeGraphResourceConstants(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                                 const vector<bool>& valid_edge_flags, RuntimeGraphResource* graph) {
    auto tensors = &graph->tensors;

    for (auto p = info.partitions.begin(); p != info.partitions.end(); ++p) {
        for (auto c = p->constants.begin(); c != p->constants.end(); ++c) {
            auto eid = c->first;
            if (!valid_edge_flags[eid]) {
                continue;
            }

            auto edge = topo->GetEdge(eid);
            if (!edge) {
                LOG(ERROR) << "cannot find edge info of constant[" << eid << "]";
                return RC_NOT_FOUND;
            }

            auto ret_pair = tensors->insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
            if (ret_pair.second) {
                auto shape_ref = info.shapes.find(eid);
                if (shape_ref == info.shapes.end()) {
                    LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }

                auto tensor = &ret_pair.first->second;
                tensor->SetBuffer(c->second.GetBufferDesc(), c->second.GetDevice());
                *tensor->GetShape() = shape_ref->second;
                graph->edgeid2object[eid] = tensor;
            }
        }
    }

    return RC_SUCCESS;
}

static void InitRuntimeGraphResourceReservedTensors(const ir::GraphTopo* topo, const set<edgeid_t>& reserved_edgeids,
                                                    RuntimeGraphResource* graph) {
    for (auto x = reserved_edgeids.begin(); x != reserved_edgeids.end(); ++x) {
        auto edge = topo->GetEdge(*x);
        auto ret_pair = graph->tensors.insert(make_pair(*x, TensorImpl(edge, TENSORTYPE_RESERVED)));
        graph->edgeid2object[*x] = &ret_pair.first->second;
    }
}

static RetCode InitRuntimeGraphResource(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                        const RuntimeInitInfo& init_info, const set<edgeid_t>& reserved_edgeids,
                                        vector<unique_ptr<EngineContext>>* engctx, RuntimeGraphResource* graph) {
    graph->nodeid2kernel.resize(topo->GetCurrentNodeIdBound());
    graph->edgeid2object.resize(topo->GetCurrentEdgeIdBound());

    auto status = InitRuntimeGraphResourceKernels(topo, info, init_info.valid_node_flags, engctx, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphResourceKernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphResourceConstants(topo, info, init_info.valid_edge_flags, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphResourceConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphResourceInputs(topo, info, init_info.name2nodeid, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphResourceInputs failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphResourceExtraInputs(topo, info, init_info.name2nodeid, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphResourceExtraInputs failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphResourceOutputs(topo, info, init_info.name2nodeid, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphResourceOutputs failed: " << GetRetCodeStr(status);
        return status;
    }

    InitRuntimeGraphResourceReservedTensors(topo, reserved_edgeids, graph);

    return RC_SUCCESS;
}

RetCode RuntimeImpl::Init(const shared_ptr<ir::GraphTopo>& topo, const shared_ptr<const RuntimeGraphInfo>& info,
                          const shared_ptr<const RuntimeAuxInfo>& aux_info, const RuntimeInitInfo& init_info,
                          const set<edgeid_t>& reserved_edgeids) {
    graph_info_ = info;
    aux_info_ = aux_info;
    topo_ = topo;

    auto status = InitRuntimeGraphResource(topo.get(), *info, init_info, reserved_edgeids, &engctx_, &graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphResource failed: " << GetRetCodeStr(status);
        return status;
    }

    sched_.reset(new SequentialScheduler());
    return sched_->Init(topo.get(), aux_info.get(), &graph_);
}

RetCode RuntimeImpl::Sync() {
    for (auto e = engctx_.begin(); e != engctx_.end(); ++e) {
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

RetCode RuntimeImpl::Run() {
    RetCode status;

    for (auto x = engctx_.begin(); x != engctx_.end(); ++x) {
        status = x->get()->Preprocess(topo_.get(), aux_info_.get(), &graph_);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "BeforeRun() of EngineContext[" << x->get()->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    Profiler* profiler = profiler_.get();
#else
    constexpr Profiler* profiler = nullptr;
#endif

    status = sched_->Run(
        [](KernelImpl* kernel, KernelExecContext* ctx) -> RetCode {
            return kernel->Execute(ctx);
        },
        profiler);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run() failed: " << GetRetCodeStr(status);
        return status;
    }

    return Sync();
}

RetCode RuntimeImpl::GetProfilingStatistics(ProfilingStatistics* stat) const {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    return profiler_->GetProfilingStatistics(stat);
#else
    LOG(ERROR) << "this version does not support profiling.";
    return RC_UNSUPPORTED;
#endif
}

Tensor* RuntimeImpl::GetTensorByName(const char* name) const {
    const string name_s(name);
    for (auto x = graph_.tensors.begin(); x != graph_.tensors.end(); ++x) {
        if (x->second.GetName() == name_s) {
            return const_cast<TensorImpl*>(&x->second);
        }
    }
    return nullptr;
}

/* -------------------------------------------------------------------------- */

RetCode RuntimeImpl::SetProfilingFlag(RuntimeImpl* rt, va_list args) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto flag = va_arg(args, uint32_t);
    bool profiling_flag = (flag > 0);

    if (profiling_flag) {
        if (!rt->profiler_) {
            rt->profiler_ = make_shared<Profiler>();
            rt->profiler_->Init(&rt->graph_, rt->aux_info_.get());
            rt->profiler_->StartProfiling(rt->topo_->GetCurrentNodeIdBound());
        }
    } else {
        if (rt->profiler_) {
            rt->profiler_->StopProfiling();
            rt->profiler_.reset();
        }
    }

    return RC_SUCCESS;
#else
    LOG(ERROR) << "this version does not support profiling.";
    return RC_UNSUPPORTED;
#endif
}

RetCode RuntimeImpl::InferShapes(RuntimeImpl* rt, va_list) {
    return rt->sched_->InferShapes();
}

RuntimeImpl::ConfHandlerFunc RuntimeImpl::conf_handlers_[] = {
    RuntimeImpl::SetProfilingFlag,
    RuntimeImpl::InferShapes,
};

RetCode RuntimeImpl::Configure(uint32_t option, ...) {
    if (option >= RUNTIME_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << RUNTIME_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}} // namespace ppl::nn
