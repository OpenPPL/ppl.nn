#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_roialign_kernel.h"

#include "cudakernel/nn/mmcv_roialign.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MMCVROIAlignKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto rois = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status =
        PPLCUDAMMCVROIAlignForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &rois->GetShape(),
                                      rois->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), *param_);
    return status;
}

}}} // namespace ppl::nn::cuda
