#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_BRIDGE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_BRIDGE_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

class BridgeOp final : public CudaOptKernel {
public:
    BridgeOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;
    ppl::common::RetCode AddInternalBridgeNode(ir::Node*, ir::Node*, ir::Edge*, ir::Graph*);
    ppl::common::RetCode AddFinalBridgeNode(ir::Node*, ir::Node*, ir::Edge*, ir::Graph*);
    ppl::common::RetCode DeleteBridgeNode(ir::Node*, ir::Graph*,
                                          std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors);
};

}}} // namespace ppl::nn::cuda

#endif
