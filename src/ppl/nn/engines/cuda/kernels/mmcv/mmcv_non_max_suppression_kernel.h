#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_MMCV_MMCV_NON_MAX_SUPPRESSION_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_MMCV_MMCV_NON_MAX_SUPPRESSION_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/params/mmcv/mmcv_non_max_suppression_param.h"

namespace ppl { namespace nn { namespace cuda {

class MMCVNonMaxSuppressionKernel : public CudaKernel {
public:
    MMCVNonMaxSuppressionKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::MMCVNMSParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::MMCVNMSParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
