#include "ppl/nn/engines/cuda/kernels/onnx/scatter_elements_kernel.h"

#include "cudakernel/memory/scatter_elements.h"

namespace ppl { namespace nn { namespace cuda {

bool ScatterElementsKernel::CanDoExecute(const KernelExecContext& ctx) const {
    return ctx.GetInput<TensorImpl>(0)->GetShape().GetBytesIncludingPadding() != 0;
}

ppl::common::RetCode ScatterElementsKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto updates = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto status = PPLCUDAScatterElementsForwardImp(
        GetStream(), &input->GetShape(), input->GetBufferPtr(), &indices->GetShape(), indices->GetBufferPtr(),
        &updates->GetShape(), updates->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), param_->axis);
    return status;
}

}}} // namespace ppl::nn::cuda
