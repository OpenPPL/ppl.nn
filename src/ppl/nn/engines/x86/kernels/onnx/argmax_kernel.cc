#include "ppl/nn/engines/x86/kernels/onnx/argmax_kernel.h"
#include "ppl/kernel/x86/fp32/argmax.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ArgMaxKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto reduced = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_X86_DEBUG_TRACE("Input [reduced]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(reduced);
    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("keepdims: %d\n", param_->keepdims);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = data->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::argmax_ndarray_fp32(&data->GetShape(), data->GetBufferPtr<float>(), param_->axis,
                                                reduced->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
