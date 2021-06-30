#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_ONNXROIALIGN_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_ONNXROIALIGN_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/roialign_param.h"

namespace ppl { namespace nn { namespace cuda {

class ONNXROIAlignKernel : public CudaKernel {
public:
    ONNXROIAlignKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::ROIAlignParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ROIAlignParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
