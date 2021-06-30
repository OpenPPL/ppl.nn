#include "ppl/nn/engines/x86/kernels/onnx/not_kernel.h"
#include "ppl/kernel/x86/bool/not.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode NotKernel::DoExecute(KernelExecContext* ctx) {
    auto X = ctx->GetInput<TensorImpl>(0);
    auto Y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (MayUseISA(ppl::common::ISA_X86_AVX)) {
        kernel::x86::not_bool_avx(&X->GetShape(), X->GetBufferPtr<uint8_t>(), Y->GetBufferPtr<uint8_t>());
    } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
        kernel::x86::not_bool_sse(&X->GetShape(), X->GetBufferPtr<uint8_t>(), Y->GetBufferPtr<uint8_t>());
    } else {
        kernel::x86::not_bool(&X->GetShape(), X->GetBufferPtr<uint8_t>(), Y->GetBufferPtr<uint8_t>());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
