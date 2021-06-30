#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_MMCV_MMCV_ROIALIGN_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_MMCV_MMCV_ROIALIGN_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/params/mmcv/mmcv_roialign_param.h"

namespace ppl { namespace nn { namespace cuda {

class MMCVROIAlignKernel : public CudaKernel {
public:
    MMCVROIAlignKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::MMCVROIAlignParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::MMCVROIAlignParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
