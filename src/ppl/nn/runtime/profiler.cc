#include "ppl/nn/runtime/profiler.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

void Profiler::Init(const RuntimeInternalConf* conf, const RuntimeGraph* graph, const RuntimeAuxInfo* aux_info) {
    conf_ = conf;
    graph_ = graph;
    aux_info_ = aux_info;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
void Profiler::CollectStatistics(KernelImpl* kernel) {
    if (conf_->profiling_flag) {
        auto info = &nodeid2info_[kernel->GetNode()->GetId()];
        info->exec_microseconds += kernel->GetExecutionTime();
        ++info->exec_count;
    }
}

void Profiler::StartProfiling(nodeid_t max_node_id) {
    nodeid2info_.resize(max_node_id);
}

RetCode Profiler::GetProfilingStatistics(ProfilingStatistics* stat) const {
    if (!conf_->profiling_flag) {
        LOG(ERROR) << "RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG is not enabled.";
        return RC_INVALID_VALUE;
    }

    stat->prof_info.reserve(aux_info_->sorted_nodes.size());
    for (auto x = aux_info_->sorted_nodes.begin(); x != aux_info_->sorted_nodes.end(); ++x) {
        auto nid = *x;
        auto& info = nodeid2info_[nid];
        auto kernel = graph_->nodeid2kernel[nid].get();

        KernelProfilingInfo kernel_prof_info;
        kernel_prof_info.name = kernel->GetName();
        auto& op_type = kernel->GetType();
        kernel_prof_info.domain = op_type.domain;
        kernel_prof_info.type = op_type.name;
        kernel_prof_info.exec_microseconds = info.exec_microseconds;
        kernel_prof_info.exec_count = info.exec_count;
        stat->prof_info.emplace_back(std::move(kernel_prof_info));
    }

    return RC_SUCCESS;
}

void Profiler::StopProfiling() {
    nodeid2info_.clear();
}
#endif

}} // namespace ppl::nn
