#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONV_HMMA_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONV_HMMA_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

namespace ppl { namespace nn { namespace cuda {

class ConvHmmaKernel : public CudaKernel {
public:
    ConvHmmaKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const CudaConvParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode BeforeExecute(KernelExecContext*) override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const CudaConvParam* param_ = nullptr;
    // ConvFuse fuse_params_;
    // BufferDesc cvt_filter_;
    // BufferDesc bias_;
};

}}} // namespace ppl::nn::cuda

#endif
