#include "ppl/nn/engines/x86/kernels/onnx/leaky_relu_kernel.h"

#include "ppl/kernel/x86/fp32/leaky_relu.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode LeakyReluKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {
            return kernel::x86::leaky_relu_fp32_avx(&y->GetShape(), x->GetBufferPtr<float>(), param_->alpha,
                                                    y->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return kernel::x86::leaky_relu_fp32_sse(&y->GetShape(), x->GetBufferPtr<float>(), param_->alpha,
                                                    y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
