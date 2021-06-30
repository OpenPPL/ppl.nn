#include "ppl/nn/engines/cuda/kernels/onnx/expand_kernel.h"

#include "cudakernel/memory/expand.h"

namespace ppl { namespace nn { namespace cuda {

bool ExpandKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto input = ctx.GetInput<TensorImpl>(0);
    auto shape = ctx.GetInput<TensorImpl>(1);
    auto output = ctx.GetOutput<TensorImpl>(0);
    if (input->GetShape().GetBytesIncludingPadding() == 0 || shape->GetShape().GetBytesIncludingPadding() == 0 ||
        output->GetShape().GetBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ExpandKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape().GetElementsIncludingPadding() == output->GetShape().GetElementsIncludingPadding()) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDAExpandForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                         output->GetBufferPtr());
    }
    return status;
}

}}} // namespace ppl::nn::cuda
