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
#include "ppl/nn/runtime/partition_runner_impl.h"
#include "ppl/nn/runtime/sequential_scheduler.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/utils/utils.h"
#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RuntimeImpl::~RuntimeImpl() {
    sched_.reset();
    nodeid2kernel_.clear();
    reserved_tensors_.clear();
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

static RetCode GenGraphKernels(const RuntimeGraphInfo& info, vector<unique_ptr<EngineContext>>* engctx,
                               vector<unique_ptr<KernelImpl>>* n2k) {
    map<EngineImpl*, EngineContext*> eng2ctx;
    for (auto partition = info.partitions.begin(); partition != info.partitions.end(); ++partition) {
        auto ctx = FindOrCreateEngineContext(partition->engine, &eng2ctx, engctx);
        if (!ctx) {
            LOG(ERROR) << "create EngineContext for engine[" << partition->engine->GetName() << "] failed.";
            return RC_OTHER_ERROR;
        }

        for (auto o = partition->ops.begin(); o != partition->ops.end(); ++o) {
            auto impl = (*o)->CreateKernelImpl();
            if (!impl) {
                LOG(ERROR) << "create kernel[" << (*o)->GetNode()->GetName() << "] failed.";
                return RC_OTHER_ERROR;
            }
            impl->SetEngineContext(ctx);
            n2k->at((*o)->GetNode()->GetId()).reset(impl);
        }
    }

    return RC_SUCCESS;
}

static RetCode GenGraphInputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                              const map<string, nodeid_t>& name2nodeid,
                              const vector<unique_ptr<KernelImpl>>& nodeid2kernel,
                              map<string, TensorImpl>* reserved_tensors) {
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        auto edge = topo->GetEdge(eid);
        auto ret_pair = reserved_tensors->insert(make_pair(edge->GetName(), TensorImpl(edge, TENSORTYPE_RESERVED)));
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
                auto kernel = nodeid2kernel[nid_ref->second].get();
                tensor->SetDevice(kernel->GetEngineContext()->GetDevice());
                break;
            }

            // ONNX supports reshaping inputs in runtime stage
            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode GenGraphExtraInputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                   const map<string, nodeid_t>& name2nodeid,
                                   const vector<unique_ptr<KernelImpl>>& nodeid2kernel,
                                   map<string, TensorImpl>* reserved_tensors) {
    for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
        auto eid = topo->GetExtraInput(i);
        auto edge = topo->GetEdge(eid);
        auto ret_pair = reserved_tensors->insert(make_pair(edge->GetName(), TensorImpl(edge, TENSORTYPE_RESERVED)));
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
                auto kernel = nodeid2kernel[nid_ref->second].get();
                tensor->SetDevice(kernel->GetEngineContext()->GetDevice());
                break;
            }

            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }
    }

    return RC_SUCCESS;
}

RetCode GenGraphOutputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                        const map<string, nodeid_t>& name2nodeid, const vector<unique_ptr<KernelImpl>>& nodeid2kernel,
                        map<string, TensorImpl>* reserved_tensors) {
    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        auto eid = topo->GetOutput(i);
        auto edge = topo->GetEdge(eid);

        auto ret_pair = reserved_tensors->insert(make_pair(edge->GetName(), TensorImpl(edge, TENSORTYPE_RESERVED)));
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

                auto kernel = nodeid2kernel[nid_ref->second].get();
                tensor->SetDevice(kernel->GetEngineContext()->GetDevice());
            }

            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode GenGraphConstants(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                 map<string, TensorImpl>* reserved_tensors) {
    for (auto p = info.partitions.begin(); p != info.partitions.end(); ++p) {
        for (auto c = p->constants.begin(); c != p->constants.end(); ++c) {
            auto eid = c->first;
            auto edge = topo->GetEdge(eid);
            if (!edge) {
                LOG(ERROR) << "cannot find edge info of constant[" << eid << "]";
                return RC_NOT_FOUND;
            }

            auto ret_pair = reserved_tensors->insert(make_pair(edge->GetName(), TensorImpl(edge, TENSORTYPE_RESERVED)));
            if (ret_pair.second) {
                auto shape_ref = info.shapes.find(eid);
                if (shape_ref == info.shapes.end()) {
                    LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }

                auto tensor = &ret_pair.first->second;
                tensor->SetBuffer(c->second.GetBufferDesc(), c->second.GetDevice());
                *tensor->GetShape() = shape_ref->second;
            }
        }
    }

    // some constants may be empty and are not in partition data
    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        auto eid = topo->GetConstant(i);
        auto edge = topo->GetEdge(eid);
        if (!edge) {
            LOG(ERROR) << "cannot find edge info of constant[" << eid << "]";
            return RC_NOT_FOUND;
        }

        reserved_tensors->insert(make_pair(edge->GetName(), TensorImpl(edge, TENSORTYPE_RESERVED)));
    }

    return RC_SUCCESS;
}

static void GenReservedTensors(const ir::GraphTopo* topo, const set<edgeid_t>& reserved_edgeids,
                               map<string, TensorImpl>* reserved_tensors) {
    for (auto x = reserved_edgeids.begin(); x != reserved_edgeids.end(); ++x) {
        auto edge = topo->GetEdge(*x);
        reserved_tensors->insert(make_pair(edge->GetName(), TensorImpl(edge, TENSORTYPE_RESERVED)));
    }
}

static RetCode InitGraphResources(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                  const RuntimeAuxInfo& aux_info, const set<edgeid_t>& reserved_edgeids,
                                  vector<unique_ptr<EngineContext>>* engctx, map<string, TensorImpl>* reserved_tensors,
                                  vector<unique_ptr<KernelImpl>>* nodeid2kernel, vector<EdgeObject*>* edgeid2object) {
    nodeid2kernel->resize(topo->GetCurrentNodeIdBound());
    auto status = GenGraphKernels(info, engctx, nodeid2kernel);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenGraphKernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenGraphConstants(topo, info, reserved_tensors);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenGraphConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenGraphInputs(topo, info, aux_info.name2nodeid, *nodeid2kernel, reserved_tensors);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenGraphInputs failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenGraphExtraInputs(topo, info, aux_info.name2nodeid, *nodeid2kernel, reserved_tensors);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenGraphExtraInputs failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenGraphOutputs(topo, info, aux_info.name2nodeid, *nodeid2kernel, reserved_tensors);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenGraphOutputs failed: " << GetRetCodeStr(status);
        return status;
    }

    GenReservedTensors(topo, reserved_edgeids, reserved_tensors);

    edgeid2object->resize(topo->GetCurrentEdgeIdBound(), nullptr);
    for (auto x = reserved_tensors->begin(); x != reserved_tensors->end(); ++x) {
        auto eid = x->second.GetEdge()->GetId();
        edgeid2object->at(eid) = &x->second;
    }

    return RC_SUCCESS;
}

RetCode RuntimeImpl::Init(const shared_ptr<ir::GraphTopo>& topo, const shared_ptr<const RuntimeGraphInfo>& info,
                          const shared_ptr<const RuntimeAuxInfo>& aux_info, const set<edgeid_t>& reserved_edgeids) {
    graph_info_ = info;
    aux_info_ = aux_info;
    topo_ = topo;

    auto status = InitGraphResources(topo.get(), *info, *aux_info, reserved_edgeids, &engctx_, &reserved_tensors_,
                                     &nodeid2kernel_, &edgeid2object_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitGraphResources failed: " << GetRetCodeStr(status);
        return status;
    }

    sched_.reset(new SequentialScheduler());
    return sched_->Init(Scheduler::Options(topo.get(), &aux_info->sorted_nodes, &aux_info->edge_last_consumer,
                                           &edgeid2object_, &nodeid2kernel_));
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
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    Profiler* profiler = profiler_.get();
#else
    constexpr Profiler* profiler = nullptr;
#endif

    auto status = sched_->Run(profiler);
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

Tensor* RuntimeImpl::GetTensor(const char* name) const {
    const string name_s(name);
    auto ref = reserved_tensors_.find(name_s);
    if (ref != reserved_tensors_.end()) {
        return const_cast<TensorImpl*>(&ref->second);
    }
    return nullptr;
}

/* -------------------------------------------------------------------------- */

static RetCode CollectPartitionEndNodeIds(const ir::GraphTopo* topo, const char** outputs, uint32_t nr_output,
                                          set<nodeid_t>* res) {
    for (uint32_t i = 0; i < nr_output; ++i) {
        auto edge = topo->GetEdge(outputs[i]);
        if (!edge) {
            LOG(ERROR) << "cannot find output[" << outputs[i] << "]";
            return RC_NOT_FOUND;
        }

        auto nid = edge->GetProducer();
        if (nid != INVALID_NODEID) {
            res->insert(nid);
        }
    }

    return RC_SUCCESS;
}

static RetCode CollectPartitionInputConsumerIds(const ir::GraphTopo* topo, const char** inputs, uint32_t nr_input,
                                                set<nodeid_t>* res) {
    for (uint32_t i = 0; i < nr_input; ++i) {
        auto edge = topo->GetEdge(inputs[i]);
        if (!edge) {
            LOG(ERROR) << "cannot find input[" << inputs[i] << "]";
            return RC_NOT_FOUND;
        }

        for (auto iter = edge->CreateConsumerIter(); iter.IsValid(); iter.Forward()) {
            auto nid = iter.Get();
            if (nid != INVALID_NODEID) {
                res->insert(nid);
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode CollectPartitionNodes(const ir::GraphTopo* topo, const char** inputs, uint32_t nr_input,
                                     const char** outputs, uint32_t nr_output, vector<nodeid_t>* nodes) {
    set<nodeid_t> end_nids;
    auto rc = CollectPartitionEndNodeIds(topo, outputs, nr_output, &end_nids);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CollectPartitionEndNodeIds failed: " << GetRetCodeStr(rc);
        return rc;
    }

    set<nodeid_t> input_consumer_nids;
    rc = CollectPartitionInputConsumerIds(topo, inputs, nr_input, &input_consumer_nids);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CollectPartitionInputConsumerIds failed: " << GetRetCodeStr(rc);
        return rc;
    }

    utils::ReversedDfs(
        topo->GetCurrentNodeIdBound(),
        [&end_nids](const function<void(nodeid_t)>& f) -> void {
            for (auto o = end_nids.begin(); o != end_nids.end(); ++o) {
                f(*o);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x = prevs.begin(); x != prevs.end(); ++x) {
                f(*x);
            }
        },
        [nodes](nodeid_t nid) -> void {
            nodes->push_back(nid);
        },
        [&input_consumer_nids](nodeid_t current) -> bool {
            return (input_consumer_nids.find(current) != input_consumer_nids.end());
        });

    return RC_SUCCESS;
}

PartitionRunner* RuntimeImpl::CreatePartitionRunner(const char** inputs, uint32_t nr_input, const char** outputs,
                                                    uint32_t nr_output) {
    vector<nodeid_t> nodes;
    auto rc = CollectPartitionNodes(topo_.get(), inputs, nr_input, outputs, nr_output, &nodes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CollectPartitionNodes failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    auto runner = new PartitionRunnerImpl();
    if (runner) {
        auto rc = runner->Init(topo_, nodes, &engctx_, &edgeid2object_, &nodeid2kernel_);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init PartitionRunner failed: " << GetRetCodeStr(rc);
            delete runner;
            return nullptr;
        }
    }

    return runner;
}

/* -------------------------------------------------------------------------- */

RetCode RuntimeImpl::ConfSetProfilingFlag(RuntimeImpl* rt, va_list args) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto flag = va_arg(args, uint32_t);
    bool profiling_flag = (flag > 0);

    if (profiling_flag) {
        if (!rt->profiler_) {
            rt->profiler_ = make_shared<Profiler>();
            rt->profiler_->Init(&rt->nodeid2kernel_, rt->aux_info_.get());
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

RetCode RuntimeImpl::ConfInferShapes(RuntimeImpl* rt, va_list) {
    return rt->sched_->ForEach([](KernelImpl* kernel, KernelExecContext* ctx) -> RetCode {
        return kernel->Reshape(ctx);
    });
}

static void DummyDeleter(Scheduler*) {}

RetCode RuntimeImpl::ConfSetScheduler(RuntimeImpl* rt, va_list args) {
    auto sched = va_arg(args, Scheduler*);
    auto rc =
        sched->Init(Scheduler::Options(rt->topo_.get(), &rt->aux_info_->sorted_nodes,
                                       &rt->aux_info_->edge_last_consumer, &rt->edgeid2object_, &rt->nodeid2kernel_));
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init user's scheduler failed: " << GetRetCodeStr(rc);
        return rc;
    }

    rt->sched_.reset(sched, DummyDeleter);
    return RC_SUCCESS;
}

RuntimeImpl::ConfHandlerFunc RuntimeImpl::conf_handlers_[] = {
    RuntimeImpl::ConfSetProfilingFlag,
    RuntimeImpl::ConfInferShapes,
    RuntimeImpl::ConfSetScheduler,
};

RetCode RuntimeImpl::Configure(uint32_t option, ...) {
    if (option >= RUNTIME_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << (uint32_t)RUNTIME_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}} // namespace ppl::nn
