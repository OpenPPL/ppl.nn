#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_PAD_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_PAD_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/pad_param.h"
#include "cudakernel/memory/pad.h"

namespace ppl { namespace nn { namespace cuda {

class PadKernel : public CudaKernel {
public:
    PadKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::PadParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::PadParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
