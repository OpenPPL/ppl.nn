#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPT_KERNEL_CREATOR_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPT_KERNEL_CREATOR_MANAGER_H_

#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

typedef X86OptKernel* (*OptKernelCreator)(const ir::Node*);

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

}}} // namespace ppl::nn::x86

#endif
