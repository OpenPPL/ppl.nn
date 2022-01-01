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
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/runtime/sequential_scheduler.h"
#include "ppl/nn/runtime/runtime_internal_conf.h"
#include "ppl/nn/utils/utils.h"
#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RuntimeImpl::~RuntimeImpl() {
    // NOTE: released before SharedResource
    sched_.reset();
    graph_.Clear();
    graph_info_.reset();
    engctx_.clear();
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

static RetCode InitRuntimeGraphKernels(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                       vector<unique_ptr<EngineContext>>* engctx, RuntimeGraph* graph) {
    graph->nodeid2kernel.resize(topo->GetMaxNodeId());

    map<EngineImpl*, EngineContext*> eng2ctx;
    for (auto partition = info.partitions.begin(); partition != info.partitions.end(); ++partition) {
        auto ctx = FindOrCreateEngineContext(partition->engine, &eng2ctx, engctx);
        if (!ctx) {
            LOG(ERROR) << "create EngineContext for engine[" << partition->engine->GetName() << "] failed.";
            return RC_OTHER_ERROR;
        }

        auto dev = ctx->GetDevice();
        for (auto o = partition->ops.begin(); o != partition->ops.end(); ++o) {
            auto impl = (*o)->CreateKernelImpl();
            if (!impl) {
                LOG(ERROR) << "create kernel[" << (*o)->GetNode()->GetName() << "] failed.";
                return RC_OTHER_ERROR;
            }
            impl->SetDevice(dev);
            graph->nodeid2kernel[(*o)->GetNode()->GetId()].reset(impl);
        }
    }

    return RC_SUCCESS;
}

static KernelImpl* FindKernelByName(const vector<unique_ptr<KernelImpl>>& kernels, const string& name) {
    for (auto it = kernels.begin(); it != kernels.end(); ++it) {
        auto kernel = it->get();
        if (kernel && kernel->GetName() == name) {
            return kernel;
        }
    }
    return nullptr;
}

static RetCode InitRuntimeGraphInputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info, RuntimeGraph* graph) {
    graph->inputs.reserve(topo->GetInputCount());

    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        auto edge = topo->GetEdgeById(eid);
        auto ret_pair = graph->tensors.insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
        auto tensor = &ret_pair.first->second;

        if (ret_pair.second) {
            // finds a consumer to get device for this input
            for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer = topo->GetNodeById(it.Get());
                if (utils::IsPplConverterNode(consumer)) {
                    continue;
                }

                // Consumers of an edge are in the same engine. This is guranteed by optimizer.
                auto kernel = FindKernelByName(graph->nodeid2kernel, consumer->GetName());
                if (!kernel) {
                    LOG(ERROR) << "cannot find consumer[" << consumer->GetName() << "] of [" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }
                tensor->SetDevice(kernel->GetDevice());
                break;
            }

            // ONNX supports reshaping inputs in runtime stage
            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }

        graph->inputs.push_back(tensor);
    }

    return RC_SUCCESS;
}

static RetCode InitRuntimeGraphExtraInputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                           RuntimeGraph* graph) {
    graph->extra_inputs.reserve(topo->GetExtraInputCount());

    for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
        auto eid = topo->GetExtraInput(i);
        auto edge = topo->GetEdgeById(eid);
        auto ret_pair = graph->tensors.insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
        auto tensor = &ret_pair.first->second;

        if (ret_pair.second) {
            // finds a consumer to get device for this extra input
            for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer = topo->GetNodeById(it.Get());
                if (utils::IsPplConverterNode(consumer)) {
                    continue;
                }

                // Consumers of an edge are in the same engine. This is guranteed by optimizer.
                auto kernel = FindKernelByName(graph->nodeid2kernel, consumer->GetName());
                if (!kernel) {
                    LOG(ERROR) << "cannot find consumer[" << consumer->GetName() << "] of [" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }
                tensor->SetDevice(kernel->GetDevice());
                break;
            }

            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }

        graph->extra_inputs.push_back(tensor);
    }

    return RC_SUCCESS;
}

RetCode InitRuntimeGraphOutputs(const ir::GraphTopo* topo, const RuntimeGraphInfo& info, RuntimeGraph* graph) {
    graph->outputs.reserve(topo->GetOutputCount());

    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        auto eid = topo->GetOutput(i);
        auto edge = topo->GetEdgeById(eid);

        auto ret_pair = graph->tensors.insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
        auto tensor = &ret_pair.first->second;

        if (ret_pair.second) {
            auto producer_id = edge->GetProducer();
            if (producer_id != INVALID_NODEID) {
                auto producer = topo->GetNodeById(producer_id);
                auto kernel = FindKernelByName(graph->nodeid2kernel, producer->GetName());
                if (!kernel) {
                    LOG(ERROR) << "cannot find producer[" << producer->GetName() << "] of [" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }
                tensor->SetDevice(kernel->GetDevice());
            }

            auto shape_ref = info.shapes.find(edge->GetId());
            if (shape_ref != info.shapes.end()) {
                *tensor->GetShape() = shape_ref->second;
            }
        }

        graph->outputs.push_back(tensor);
    }

    return RC_SUCCESS;
}

static RetCode InitRuntimeGraphConstants(const ir::GraphTopo* topo, const RuntimeGraphInfo& info, RuntimeGraph* graph) {
    auto constants = &graph->constants;
    auto tensors = &graph->tensors;

    constants->reserve(topo->GetConstantCount());

    for (auto p = info.partitions.begin(); p != info.partitions.end(); ++p) {
        for (auto c = p->constants.begin(); c != p->constants.end(); ++c) {
            auto eid = c->first;
            auto edge = topo->GetEdgeById(eid);
            if (!edge) {
                LOG(ERROR) << "cannot find edge info of constant[" << eid << "]";
                return RC_NOT_FOUND;
            }

            auto ret_pair = tensors->insert(make_pair(eid, TensorImpl(edge, TENSORTYPE_RESERVED)));
            if (ret_pair.second) {
                auto tensor = &ret_pair.first->second;
                tensor->SetBuffer(c->second.GetBufferDesc(), c->second.GetDevice());
                *tensor->GetShape() = *c->second.GetShape();
                constants->push_back(tensor);
            }
        }
    }

    return RC_SUCCESS;
}

RetCode RuntimeImpl::InitRuntimeGraph(const ir::GraphTopo* topo, const RuntimeGraphInfo& info, RuntimeGraph* graph) {
    auto status = InitRuntimeGraphKernels(topo, info, &engctx_, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphKernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphConstants(topo, info, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphInputs(topo, info, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphInputs failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphExtraInputs(topo, info, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphExtraInputs failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitRuntimeGraphOutputs(topo, info, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraphOutputs failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode RuntimeImpl::Init(const shared_ptr<ir::GraphTopo>& topo, const shared_ptr<const RuntimeGraphInfo>& info,
                          const shared_ptr<const RuntimeAuxInfo>& aux_info,
                          const shared_ptr<utils::SharedResource>& resource) {
    resource_ = resource;
    graph_info_ = info;
    aux_info_ = aux_info;
    topo_ = topo;

    profiler_.Init(&conf_, &graph_, aux_info.get());

    auto status = InitRuntimeGraph(topo.get(), *info, &graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitRuntimeGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    sched_.reset(new SequentialScheduler());
    return sched_->Init(topo.get(), aux_info.get(), &graph_);
}

RetCode RuntimeImpl::Sync() {
    for (uint32_t i = 0; i < GetOutputCount(); ++i) {
        auto output = GetOutputTensorImpl(i);
        auto barrier = output->GetBarrier();
        if (barrier) {
            auto status = barrier->Sync();
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "sync tensor[" << output->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }
    return RC_SUCCESS;
}

RetCode RuntimeImpl::Run() {
    RetCode status;

    for (auto x = engctx_.begin(); x != engctx_.end(); ++x) {
        status = x->get()->BeforeRun();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "BeforeRun() of EngineContext[" << x->get()->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    status = sched_->Run(&profiler_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run() failed: " << GetRetCodeStr(status);
        return status;
    }

    return Sync();
}

RetCode RuntimeImpl::GetProfilingStatistics(ProfilingStatistics* stat) const {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    return profiler_.GetProfilingStatistics(stat);
#else
    LOG(ERROR) << "this version does not support profiling.";
    return RC_UNSUPPORTED;
#endif
}

/* -------------------------------------------------------------------------- */

RetCode RuntimeImpl::SetProfilingFlag(RuntimeImpl* rt, va_list args) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto flag = va_arg(args, uint32_t);
    bool profiling_flag = (flag > 0);
    rt->conf_.profiling_flag = profiling_flag;

    if (profiling_flag) {
        rt->profiler_.StartProfiling(rt->topo_->GetMaxNodeId());
    } else {
        rt->profiler_.StopProfiling();
    }

    return RC_SUCCESS;
#else
    LOG(ERROR) << "this version does not support profiling.";
    return RC_UNSUPPORTED;
#endif
}

RuntimeImpl::ConfHandlerFunc RuntimeImpl::conf_handlers_[] = {
    RuntimeImpl::SetProfilingFlag,
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
