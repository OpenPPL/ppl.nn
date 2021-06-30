#include "ppl/nn/engines/x86/kernels/onnx/softmax_kernel.h"

#include "ppl/kernel/x86/fp32/softmax.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SoftmaxKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = input->GetShape().GetDataType();
    const auto data_format = input->GetShape().GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            if (MayUseISA(ppl::common::ISA_X86_FMA)) {
                return ppl::kernel::x86::softmax_ndarray_fp32_fma(&input->GetShape(), input->GetBufferPtr<float>(),
                                                                  param_->axis, output->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return ppl::kernel::x86::softmax_ndarray_fp32_sse(&input->GetShape(), input->GetBufferPtr<float>(),
                                                                  param_->axis, output->GetBufferPtr<float>());
            } else {
                return ppl::kernel::x86::softmax_ndarray_fp32(&input->GetShape(), input->GetBufferPtr<float>(),
                                                              param_->axis, output->GetBufferPtr<float>());
            }
        } else {
            LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format " << ppl::common::GetDataFormatStr(data_format) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
