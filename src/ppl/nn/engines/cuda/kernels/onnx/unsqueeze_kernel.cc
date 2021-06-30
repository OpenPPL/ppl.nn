#include "ppl/nn/engines/cuda/kernels/onnx/unsqueeze_kernel.h"

#include "cudakernel/memory/unsqueeze.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode UnsqueezeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDAUnsqueezeForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                            output->GetBufferPtr());
    }

    return status;
}

}}} // namespace ppl::nn::cuda
