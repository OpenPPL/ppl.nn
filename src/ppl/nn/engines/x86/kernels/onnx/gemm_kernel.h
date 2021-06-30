#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_GEMM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_GEMM_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/gemm_param.h"

namespace ppl { namespace nn { namespace x86 {

class GemmKernel : public X86Kernel {
public:
    GemmKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::GemmParam* p) {
        param_ = p;
    }
    void SetFuseReLU(bool fuse_relu) {
        gemm_fuse_relu_ = fuse_relu;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::GemmParam* param_ = nullptr;
    bool gemm_fuse_relu_ = false;
};

}}} // namespace ppl::nn::x86

#endif
