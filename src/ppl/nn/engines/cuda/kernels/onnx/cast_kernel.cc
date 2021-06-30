#include "ppl/nn/engines/cuda/kernels/onnx/cast_kernel.h"

#include "cudakernel/unary/cast.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode CastKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape().GetDataType() == output->GetShape().GetDataType()) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDACastForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                       output->GetBufferPtr(), param_->to);
    }

    return status;
}

}}} // namespace ppl::nn::cuda
