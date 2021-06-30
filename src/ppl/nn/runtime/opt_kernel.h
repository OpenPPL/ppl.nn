#ifndef _ST_HPC_PPL_NN_RUNTIME_OPT_KERNEL_H_
#define _ST_HPC_PPL_NN_RUNTIME_OPT_KERNEL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/kernel_impl.h"

namespace ppl { namespace nn {

/**
   @class OptKernel
   @brief used to store shared runtime data.
*/
class OptKernel {
public:
    /** @param node the corresponding node in ir::GraphTopo */
    OptKernel(const ir::Node* node) : node_(node) {}

    virtual ~OptKernel() {}

    /** @brief get the corresponding node in ir::GraphTopo */
    const ir::Node* GetNode() const {
        return node_;
    }

    /** @brief create a KernelImpl used in runtime stage */
    virtual KernelImpl* CreateKernelImpl() const = 0;

private:
    const ir::Node* node_;
};

}} // namespace ppl::nn

#endif
