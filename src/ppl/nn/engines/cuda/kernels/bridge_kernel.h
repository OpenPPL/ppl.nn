#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_BRIDGE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_BRIDGE_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

namespace ppl { namespace nn { namespace cuda {

class BridgeKernel : public CudaKernel {
public:
    BridgeKernel(const ir::Node* node) : CudaKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool EqualTypeAndFormat(const TensorImpl*, const TensorImpl*);
};

}}} // namespace ppl::nn::cuda

#endif
