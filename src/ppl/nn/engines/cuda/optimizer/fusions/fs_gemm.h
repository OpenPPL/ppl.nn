#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_GEMM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_GEMM_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"

namespace ppl { namespace nn { namespace cuda {

class GemmFusion : public Fusion {
public:
    const ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, OptKernelOptions& options) override;

private:
    const bool CanFuse(ir::Node* nextnode, OptKernelOptions& options, uint32_t flag);
    const ppl::common::RetCode FuseGemmWithNextNode(ir::Node* node, ir::Node* nextnode, OptKernelOptions& options);

private:
    std::set<std::string> fuse_type{"Relu",
                                    "Clip"
                                    "Sigmoid"};
};

}}} // namespace ppl::nn::cuda

#endif