#include "ppl/nn/engines/cuda/kernels/onnx/pad_kernel.h"

#include <memory>

#include "cudakernel/memory/pad.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode PadKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto pads = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PadKernelParam kernel_param;

    kernel_param.mode = param_->mode;
    if (ctx->GetInputCount() >= 3) {
        auto constant_data = ctx->GetInput<TensorImpl>(2);
        auto status = constant_data->CopyToHost(&(kernel_param.constant_value));
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy constant value failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape().GetElementsIncludingPadding() == output->GetShape().GetElementsIncludingPadding()) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDAPadForwardImp(GetStream(), kernel_param, &input->GetShape(), input->GetBufferPtr(),
                                      &pads->GetShape(), pads->GetBufferPtr<int64_t>(), &output->GetShape(),
                                      output->GetBufferPtr());
    }
    return status;
}

}}} // namespace ppl::nn::cuda
