#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_gridsample_kernel.h"

#include "cudakernel/nn/mmcv_gridsample.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MMCVGridSampleKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status =
        PPLCUDAMMCVGridSampleForwardImp(GetStream(), &input0->GetShape(), input0->GetBufferPtr(), &input1->GetShape(),
                                        input1->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), *param_);
    return status;
}

}}} // namespace ppl::nn::cuda
