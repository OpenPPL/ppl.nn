#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_IMPL_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_IMPL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/runtime/runtime_graph.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/runtime/runtime_options.h"
#include "ppl/nn/runtime/runtime_internal_conf.h"
#include "ppl/nn/runtime/scheduler.h"
#include "ppl/nn/utils/shared_resource.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/runtime/profiler.h"

namespace ppl { namespace nn {

class RuntimeImpl final : public Runtime {
public:
    RuntimeImpl() {}
    ~RuntimeImpl();

    ppl::common::RetCode Init(const RuntimeOptions& options, const std::shared_ptr<ir::GraphTopo>&,
                              const std::shared_ptr<const RuntimeGraphInfo>&,
                              const std::shared_ptr<const RuntimeAuxInfo>&,
                              const std::shared_ptr<utils::SharedResource>&);

    TensorImpl* GetInputTensorImpl(uint32_t idx) const {
        return graph_.inputs[idx];
    }
    TensorImpl* GetOutputTensorImpl(uint32_t idx) const {
        return graph_.outputs[idx];
    }

    uint32_t GetExtraInputCount() const {
        return graph_.extra_inputs.size();
    }
    TensorImpl* GetExtraInputTensorImpl(uint32_t idx) const {
        return graph_.extra_inputs[idx];
    }

    // ----- //

    ppl::common::RetCode Configure(uint32_t, ...) override;

    uint32_t GetInputCount() const override {
        return graph_.inputs.size();
    }
    Tensor* GetInputTensor(uint32_t idx) const override {
        return GetInputTensorImpl(idx);
    }

    uint32_t GetOutputCount() const override {
        return graph_.outputs.size();
    }
    Tensor* GetOutputTensor(uint32_t idx) const override {
        return GetOutputTensorImpl(idx);
    }

    ppl::common::RetCode Run() override;
    ppl::common::RetCode Sync() override;

    ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics* stat) const override;

private:
    ppl::common::RetCode InitRuntimeGraph(const ir::GraphTopo*, const RuntimeGraphInfo&, const RuntimeOptions&,
                                          RuntimeGraph*);

private:
    RuntimeGraph graph_;
    std::unique_ptr<Scheduler> sched_;
    std::vector<std::unique_ptr<EngineContext>> engctx_;
    utils::GenericCpuDevice cpu_device_; // default cpu device
    RuntimeInternalConf conf_;
    Profiler profiler_;

    // ----- shared data ----- //

    std::shared_ptr<ir::GraphTopo> topo_;
    std::shared_ptr<const RuntimeGraphInfo> graph_info_;
    std::shared_ptr<const RuntimeAuxInfo> aux_info_;
    std::shared_ptr<utils::SharedResource> resource_;

private:
    /*
      some of them may visit class members.
      defined as member functions can avoid exporting unnecessary APIs
    */
    static ppl::common::RetCode SetProfilingFlag(RuntimeImpl*, va_list);

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
