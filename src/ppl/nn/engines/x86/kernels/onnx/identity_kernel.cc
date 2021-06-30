#include "ppl/nn/engines/x86/kernels/onnx/identity_kernel.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode IdentityKernel::DoExecute(KernelExecContext* ctx) {
    auto input_tensor = ctx->GetInput<TensorImpl>(0);
    auto output_tensor = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("identity_kernel bottom: %p top: %p\n", input_tensor, output_tensor);
    PPLNN_X86_DEBUG_TRACE("name: %s\n", input_tensor->GetName());
    PPLNN_X86_DEBUG_TRACE("ptr: %p\n", input_tensor->GetBufferPtr());
    PPLNN_X86_DEBUG_TRACE("dims: %d %d %d %d\n", (int)input_tensor->GetShape().GetDim(0),
                          (int)input_tensor->GetShape().GetDim(1), (int)input_tensor->GetShape().GetDim(2),
                          (int)input_tensor->GetShape().GetDim(3));
    PPLNN_X86_DEBUG_TRACE("dataType: %s\n", ppl::common::GetDataTypeStr(input_tensor->GetShape().GetDataType()));
    PPLNN_X86_DEBUG_TRACE("dataFormat: %s\n", ppl::common::GetDataFormatStr(input_tensor->GetShape().GetDataFormat()));
    PPLNN_X86_DEBUG_TRACE("name: %s\n", output_tensor->GetName());
    PPLNN_X86_DEBUG_TRACE("ptr: %p\n", output_tensor->GetBufferPtr());
    PPLNN_X86_DEBUG_TRACE("dims: %d %d %d %d\n", (int)output_tensor->GetShape().GetDim(0),
                          (int)output_tensor->GetShape().GetDim(1), (int)output_tensor->GetShape().GetDim(2),
                          (int)output_tensor->GetShape().GetDim(3));
    PPLNN_X86_DEBUG_TRACE("dataType: %s\n", ppl::common::GetDataTypeStr(output_tensor->GetShape().GetDataType()));
    PPLNN_X86_DEBUG_TRACE("dataFormat: %s\n", ppl::common::GetDataFormatStr(output_tensor->GetShape().GetDataFormat()));

    memcpy(output_tensor->GetBufferPtr(), input_tensor->GetBufferPtr(),
           input_tensor->GetShape().GetBytesIncludingPadding());

    return 0;
}

}}} // namespace ppl::nn::x86
