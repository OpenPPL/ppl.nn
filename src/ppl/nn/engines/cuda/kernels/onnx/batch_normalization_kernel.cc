#include "ppl/nn/engines/cuda/kernels/onnx/batch_normalization_kernel.h"

#include "cudakernel/nn/batch_normalization.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode BatchNormalizationKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto scale = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDABatchNormalizationForwardImp(
        GetStream(), &input->GetShape(), input->GetBufferPtr(), &scale->GetShape(), scale->GetBufferPtr(),
        ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), ctx->GetInput<TensorImpl>(3)->GetBufferPtr(),
        ctx->GetInput<TensorImpl>(4)->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), param_->epsilon);
    return status;
}

}}} // namespace ppl::nn::cuda
