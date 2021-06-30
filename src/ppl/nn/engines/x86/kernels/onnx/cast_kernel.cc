#include "ppl/nn/engines/x86/kernels/onnx/cast_kernel.h"

#include "ppl/kernel/x86/common/cast.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode CastKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("to: %d\n", param_->to);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    return kernel::x86::cast(&input->GetShape(), &output->GetShape(), input->GetBufferPtr(), output->GetBufferPtr());
}

}}} // namespace ppl::nn::x86
