#include "ppl/nn/engines/cuda/kernels/onnx/gather_kernel.h"

#include "cudakernel/memory/gather.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode GatherKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape().GetDim(param_->axis) == 1 &&
        input->GetShape().GetElementsIncludingPadding() == output->GetShape().GetElementsIncludingPadding()) {
        output->TransferBufferFrom(input);
    } else {
        status =
            PPLCUDAGatherForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &indices->GetShape(),
                                    indices->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), param_->axis);
    }

    return status;
}

}}} // namespace ppl::nn::cuda
