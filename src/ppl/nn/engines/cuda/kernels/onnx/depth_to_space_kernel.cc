#include "ppl/nn/engines/cuda/kernels/onnx/depth_to_space_kernel.h"

#include "cudakernel/memory/depth_to_space.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode DepthToSpaceKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDADepthToSpaceForwardImp(
        GetStream(), *param_, &input->GetShape(), input->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
