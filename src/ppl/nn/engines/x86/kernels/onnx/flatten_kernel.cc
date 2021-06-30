#include "ppl/nn/engines/x86/kernels/onnx/flatten_kernel.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode FlattenKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL) {
        output->TransferBufferFrom(input);
    } else {
        memcpy(output->GetBufferPtr(), input->GetBufferPtr(), input->GetShape().GetBytesIncludingPadding());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
