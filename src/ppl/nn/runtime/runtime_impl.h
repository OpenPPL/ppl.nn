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

#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_IMPL_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_IMPL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/runtime/scheduler.h"
#include "ppl/nn/runtime/profiler.h"
#include "ppl/nn/runtime/options.h"

#include <cstdarg>

namespace ppl { namespace nn {

class RuntimeImpl final : public Runtime {
public:
    RuntimeImpl() {}
    ~RuntimeImpl();

    ppl::common::RetCode Init(const std::shared_ptr<ir::GraphTopo>&, const std::shared_ptr<const RuntimeGraphInfo>&,
                              const std::shared_ptr<const RuntimeAuxInfo>&,
                              const std::set<edgeid_t>& reserved_edges = {});

    TensorImpl* GetInputTensorImpl(uint32_t idx) const {
        auto eid = topo_->GetInput(idx);
        return static_cast<TensorImpl*>(edgeid2object_[eid]);
    }
    TensorImpl* GetOutputTensorImpl(uint32_t idx) const {
        auto eid = topo_->GetOutput(idx);
        return static_cast<TensorImpl*>(edgeid2object_[eid]);
    }

    uint32_t GetExtraInputCount() const {
        return topo_->GetExtraInputCount();
    }
    TensorImpl* GetExtraInputTensorImpl(uint32_t idx) const {
        auto eid = topo_->GetExtraInput(idx);
        return static_cast<TensorImpl*>(edgeid2object_[eid]);
    }

    // ----- //

    ppl::common::RetCode Configure(uint32_t, ...) override;

    PartitionRunner* CreatePartitionRunner(const char** inputs, uint32_t nr_input, const char** outputs,
                                           uint32_t nr_output) override;

    uint32_t GetInputCount() const override {
        return topo_->GetInputCount();
    }
    Tensor* GetInputTensor(uint32_t idx) const override {
        auto eid = topo_->GetInput(idx);
        return static_cast<TensorImpl*>(edgeid2object_[eid]);
    }

    uint32_t GetOutputCount() const override {
        return topo_->GetOutputCount();
    }
    Tensor* GetOutputTensor(uint32_t idx) const override {
        auto eid = topo_->GetOutput(idx);
        return static_cast<TensorImpl*>(edgeid2object_[eid]);
    }

    Tensor* GetTensor(const char* name) const override;

    ppl::common::RetCode Run() override;
    ppl::common::RetCode RunAsync() override;

    ppl::common::RetCode Synchronize() override;

    uint32_t GetDeviceContextCount() const override {
        return engctx_.size();
    }
    DeviceContext* GetDeviceContext(uint32_t idx) const override {
        return engctx_[idx]->GetDevice();
    }

    ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics* stat) const override;

private:
    std::shared_ptr<Scheduler> sched_;

    /** union of inputs/extra_inputs/constants/outputs/reserved_edgeids */
    std::map<std::string, TensorImpl> reserved_tensors_;

    /** kernels list where the subscriptor is KernelImpl::GetNode()::GetId() */
    std::vector<std::unique_ptr<KernelImpl>> nodeid2kernel_;

    /** objects that are used during Run() */
    std::vector<EdgeObject*> edgeid2object_;

    /** `EngineContext` instances of this runtime */
    std::vector<std::unique_ptr<EngineContext>> engctx_;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    std::shared_ptr<Profiler> profiler_;
#endif

    // ----- shared data ----- //

    std::shared_ptr<ir::GraphTopo> topo_;
    std::shared_ptr<const RuntimeAuxInfo> aux_info_;
    std::shared_ptr<const RuntimeGraphInfo> graph_info_;

private:
    /*
      some of them may visit class members.
      defined as member functions can avoid exporting unnecessary APIs
    */
    static ppl::common::RetCode ConfSetProfilingFlag(RuntimeImpl*, va_list);
    static ppl::common::RetCode ConfInferShapes(RuntimeImpl*, va_list);
    static ppl::common::RetCode ConfSetScheduler(RuntimeImpl*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(RuntimeImpl*, va_list);
    static ConfHandlerFunc conf_handlers_[RUNTIME_CONF_MAX];

private:
    RuntimeImpl(const RuntimeImpl&) = delete;
    RuntimeImpl& operator=(const RuntimeImpl&) = delete;
    RuntimeImpl(RuntimeImpl&&) = delete;
    RuntimeImpl& operator=(RuntimeImpl&&) = delete;
};

}} // namespace ppl::nn

#endif
