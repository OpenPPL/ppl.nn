#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPT_KERNEL_CREATOR_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPT_KERNEL_CREATOR_MANAGER_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

typedef CudaOptKernel* (*OptKernelCreator)(const ir::Node*);

class OptKernelCreatorManager {
public:
    static OptKernelCreatorManager* Instance() {
        static OptKernelCreatorManager mgr;
        return &mgr;
    }

    ppl::common::RetCode Register(const std::string& domain, const std::string& type, OptKernelCreator);
    OptKernelCreator Find(const std::string& domain, const std::string& type);

private:
    std::map<std::string, std::map<std::string, OptKernelCreator>> domain_type_creator_;

private:
    OptKernelCreatorManager();
};

}}} // namespace ppl::nn::cuda

#endif
