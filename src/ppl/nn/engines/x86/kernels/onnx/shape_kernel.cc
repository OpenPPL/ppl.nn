#include "ppl/nn/engines/x86/kernels/onnx/shape_kernel.h"

namespace ppl { namespace nn { namespace x86 {

bool ShapeKernel::CanDoExecute(const KernelExecContext&) const {
    return true;
}

ppl::common::RetCode ShapeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto shape = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_X86_DEBUG_TRACE("Output [shape]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(shape);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    for (size_t i = 0; i < data->GetShape().GetRealDimCount(); i++) {
        shape->GetBufferPtr<int64_t>()[i] = data->GetShape().GetDim(i);
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
