#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_FUSION_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_FUSION_H_

#include <set>
#include <map>
#include <vector>
#include <string>

#include "ppl/common/types.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

class Fusion {
public:
    virtual const ppl::common::RetCode FuseNode(ir::Node*, bool, OptKernelOptions&) = 0;
};

}}} // namespace ppl::nn::cuda

#endif
