#ifndef _ST_HPC_PPL_NN_UTILS_CPU_TIMING_GUARD_H_
#define _ST_HPC_PPL_NN_UTILS_CPU_TIMING_GUARD_H_

#include <chrono>

namespace ppl { namespace nn { namespace utils {

class CpuTimingGuard final {
public:
    CpuTimingGuard(std::chrono::time_point<std::chrono::system_clock>* begin_ts,
                   std::chrono::time_point<std::chrono::system_clock>* end_ts, bool is_profiling_enabled)
        : is_profiling_enabled_(is_profiling_enabled), end_ts_(end_ts) {
        if (is_profiling_enabled) {
            *begin_ts = std::chrono::system_clock::now();
        }
    }
    ~CpuTimingGuard() {
        if (is_profiling_enabled_) {
            *end_ts_ = std::chrono::system_clock::now();
        }
    }

private:
    bool is_profiling_enabled_;
    std::chrono::time_point<std::chrono::system_clock>* end_ts_;
};

}}} // namespace ppl::nn::utils

#endif
