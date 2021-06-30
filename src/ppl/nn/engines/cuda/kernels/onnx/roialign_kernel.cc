#include "ppl/nn/engines/cuda/kernels/onnx/roialign_kernel.h"

#include "cudakernel/nn/roialign.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ROIAlignKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto rois = ctx->GetInput<TensorImpl>(1);
    auto batch_indices = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status =
        PPLCUDAROIAlignForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &rois->GetShape(),
                                  rois->GetBufferPtr(), &batch_indices->GetShape(), batch_indices->GetBufferPtr(),
                                  &output->GetShape(), output->GetBufferPtr(), *param_);
    return status;
}

}}} // namespace ppl::nn::cuda
