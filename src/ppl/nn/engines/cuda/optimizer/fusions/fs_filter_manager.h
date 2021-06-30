
#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_FS_FILTER_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_FS_FILTER_MANAGER_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_averagepool.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_cast.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_concat.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_conv.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_gemm.h"

namespace ppl { namespace nn { namespace cuda {

class FsFilterManager {
public:
    static FsFilterManager* Instance() {
        static FsFilterManager mgr;
        return &mgr;
    }

    Fusion* FindFusion(const std::string& kernel_type) const;

private:
    FsFilterManager();

private:
    std::map<std::string, Fusion*> type2fusion_;
    AveragePoolFusion averagepool_fs_;
    CastFusion cast_fs_;
    ConcatFusion concat_fs_;
    ConvFusion conv_fs_;
    GemmFusion gemm_fs_;
};

}}} // namespace ppl::nn::cuda

#endif
