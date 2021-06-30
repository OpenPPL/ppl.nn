#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CONCAT_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CONCAT_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"

namespace ppl { namespace nn { namespace cuda {

class ConcatFusion : public Fusion {
public:
    const ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, OptKernelOptions& options) override;

private:
    const bool CanFuse(ir::Node* prenode, OptKernelOptions& options, uint32_t sumc);
};

}}} // namespace ppl::nn::cuda

#endif