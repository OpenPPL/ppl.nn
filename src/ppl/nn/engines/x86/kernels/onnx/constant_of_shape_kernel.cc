#include "ppl/nn/engines/x86/kernels/onnx/constant_of_shape_kernel.h"

#include "ppl/kernel/x86/common/memset_nbytes.h"

#include <chrono>

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ConstantOfShapeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    uint64_t output_datatype_size = ppl::common::GetSizeOfDataType(output->GetShape().GetDataType());
    return kernel::x86::memset_nbytes(param_->data.data(), output_datatype_size,
                                      output->GetShape().GetElementsIncludingPadding(), output->GetBufferPtr());
}

}}} // namespace ppl::nn::x86
