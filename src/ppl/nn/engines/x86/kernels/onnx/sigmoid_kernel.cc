#include "ppl/nn/engines/x86/kernels/onnx/sigmoid_kernel.h"
#include "ppl/kernel/x86/fp32/sigmiod.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SigmoidKernel::DoExecute(KernelExecContext* ctx) {
    auto X = ctx->GetInput<TensorImpl>(0);
    auto Y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = X->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_FMA)) {
            return ppl::kernel::x86::sigmoid_fp32_fma(&X->GetShape(), X->GetBufferPtr<float>(),
                                                      Y->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return ppl::kernel::x86::sigmoid_fp32_sse(&X->GetShape(), X->GetBufferPtr<float>(),
                                                      Y->GetBufferPtr<float>());
        } else {
            return ppl::kernel::x86::sigmoid_fp32(&X->GetShape(), X->GetBufferPtr<float>(), Y->GetBufferPtr<float>());
        }
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
