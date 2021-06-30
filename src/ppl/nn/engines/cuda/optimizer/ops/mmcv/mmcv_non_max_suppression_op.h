#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_MMCV_MMCV_NON_MAX_SUPPRESSION_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_MMCV_MMCV_NON_MAX_SUPPRESSION_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/mmcv/mmcv_non_max_suppression_param.h"

namespace ppl { namespace nn { namespace cuda {

class MMCVNonMaxSupressionOp final : public CudaOptKernel {
public:
    MMCVNonMaxSupressionOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

private:
    ppl::nn::common::MMCVNMSParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
