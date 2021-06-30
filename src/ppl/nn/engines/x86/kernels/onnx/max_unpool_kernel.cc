#include "ppl/nn/engines/x86/kernels/onnx/max_unpool_kernel.h"

#include "ppl/kernel/x86/fp32/max_unpool.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode MaxUnpoolKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (x->GetShape().GetDimCount() != 4 || indices->GetShape().GetDimCount() != 4) {
        LOG(ERROR) << "only support 4-D Tensor now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const auto data_type = x->GetShape().GetDataType();
    const auto data_format = x->GetShape().GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            return ppl::kernel::x86::max_unpool_nchw_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                          indices->GetBufferPtr<int64_t>(), y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format " << ppl::common::GetDataFormatStr(data_format) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
