#include "ppl/nn/engines/cuda/optimizer/algos/algo_filter_manager.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const AlgoFilter* AlgoFilterManager::FindKernel(const std::string& type) const {
    auto ref = type2algo_.find(type);
    if (ref == type2algo_.end()) {
        return nullptr;
    }
    return &ref->second;
}

#define REGISTER_ALGO_FILTER_INFO(impl, classname) \
    do {                                           \
        type2algo_[impl].AppendAlgo(&classname);   \
    } while (0)

AlgoFilterManager::AlgoFilterManager() {
    REGISTER_ALGO_FILTER_INFO("Conv", turing_hmma_imp_);
    REGISTER_ALGO_FILTER_INFO("Conv", depthwise_direct_imp_);
    REGISTER_ALGO_FILTER_INFO("Bridge", bridge_imp_);
    REGISTER_ALGO_FILTER_INFO("Concat", concat_imp_);
    REGISTER_ALGO_FILTER_INFO("Gemm", gemm_imp_);
    REGISTER_ALGO_FILTER_INFO("Normal", normal_imp_);
}

}}} // namespace ppl::nn::cuda
