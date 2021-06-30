#include "ppl/nn/engines/x86/kernels/onnx/squeeze_kernel.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SqueezeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto squeezed = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_X86_DEBUG_TRACE("Output [squeezed]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(squeezed);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (data->GetEdge()->CalcConsumerCount() == 1 && data->GetType() == TENSORTYPE_NORMAL) {
        squeezed->TransferBufferFrom(data);
    } else {
        memcpy(squeezed->GetBufferPtr(), data->GetBufferPtr(), data->GetShape().GetBytesIncludingPadding());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
