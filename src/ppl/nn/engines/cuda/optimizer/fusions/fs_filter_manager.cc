#include "ppl/nn/engines/cuda/optimizer/fusions/fs_filter_manager.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

Fusion* FsFilterManager::FindFusion(const std::string& kernel_type) const {
    auto ref = type2fusion_.find(kernel_type);
    if (ref == type2fusion_.end()) {
        return nullptr;
    }
    return ref->second;
}

FsFilterManager::FsFilterManager() {
    type2fusion_.emplace("AveragePool", &averagepool_fs_);
    type2fusion_.emplace("Cast", &cast_fs_);
    type2fusion_.emplace("Concat", &concat_fs_);
    type2fusion_.emplace("Conv", &conv_fs_);
    type2fusion_.emplace("Gemm", &gemm_fs_);
}

}}} // namespace ppl::nn::cuda
