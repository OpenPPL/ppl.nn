#ifndef _ST_HPC_PPL_NN_RUNTIME_KERNEL_EXEC_CONTEXT_H_
#define _ST_HPC_PPL_NN_RUNTIME_KERNEL_EXEC_CONTEXT_H_

#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn {

/**
   @class KernelExecContext
   @brief kernel execution context
*/
class KernelExecContext final : public InputOutputInfo {
public:
    void SetProfilingFlag(bool is_profiling_enabled) {
        is_profiling_enabled_ = is_profiling_enabled;
    }
    bool IsProfilingEnabled() const {
        return is_profiling_enabled_;
    }

    void SetGetBarrierFunc(const std::function<Barrier*(edgeid_t)>& f) {
        get_barrier_func_ = f;
    }

    Barrier* GetInputBarrier(uint32_t idx) const {
        auto eid = node_->GetInput(idx);
        return get_barrier_func_(eid);
    }

    Barrier* GetExtraInputBarrier(uint32_t idx) const {
        auto eid = node_->GetExtraInput(idx);
        return get_barrier_func_(eid);
    }

    Barrier* GetOutputBarrier(uint32_t idx) const {
        auto eid = node_->GetOutput(idx);
        return get_barrier_func_(eid);
    }

private:
    bool is_profiling_enabled_ = false;
    std::function<Barrier*(edgeid_t)> get_barrier_func_;
};

}} // namespace ppl::nn

#endif
