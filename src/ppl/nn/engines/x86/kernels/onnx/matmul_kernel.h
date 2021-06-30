#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_MATMUL_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_MATMUL_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"

namespace ppl { namespace nn { namespace x86 {

class MatMulKernel : public X86Kernel {
public:
    MatMulKernel(const ir::Node* node) : X86Kernel(node) {}

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
};

}}} // namespace ppl::nn::x86

#endif
