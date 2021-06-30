#ifndef _ST_HPC_PPL_NN_RUNTIME_PROFILER_H_
#define _ST_HPC_PPL_NN_RUNTIME_PROFILER_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/runtime_internal_conf.h"
#include "ppl/nn/runtime/runtime_graph.h"
#include "ppl/nn/runtime/runtime_aux_info.h"

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include "ppl/nn/runtime/profiling_statistics.h"
#include <chrono>
#endif

namespace ppl { namespace nn {

class Profiler final {
public:
    void Init(const RuntimeInternalConf* conf, const RuntimeGraph* graph, const RuntimeAuxInfo* aux_info);

    bool IsProfilingEnabled() const {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        return conf_->profiling_flag;
#else
        return false;
#endif
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    void CollectStatistics(KernelImpl*);

public:
    void StartProfiling(nodeid_t max_node_id);
    ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics*) const;
    void StopProfiling();

private:
    struct KernelExecInfo {
        uint32_t exec_count = 0;
        uint64_t exec_microseconds = 0;
    };

    std::vector<KernelExecInfo> nodeid2info_;
#endif

private:
    const RuntimeInternalConf* conf_;
    const RuntimeGraph* graph_;
    const RuntimeAuxInfo* aux_info_;
};

}} // namespace ppl::nn

#endif
