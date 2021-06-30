#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_PPL_REORDER_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_PPL_REORDER_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"

namespace ppl { namespace nn { namespace x86 {

inline TensorShape PadShapeTo3Dims(const TensorShape& shape) {
    if (shape.GetDimCount() >= 3) {
        return shape;
    }

    std::vector<int64_t> padded_dims(3, 1);
    const uint32_t offset = 3 - shape.GetRealDimCount();
    for (uint32_t i = offset; i < padded_dims.size(); i++) {
        padded_dims[i] = shape.GetDim(i - offset);
    }

    TensorShape padded_shape(shape);
    padded_shape.Reshape(padded_dims);
    return padded_shape;
}

class ReorderKernel : public X86Kernel {
public:
    ReorderKernel(const ir::Node* node) : X86Kernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
};

}}} // namespace ppl::nn::x86

#endif
