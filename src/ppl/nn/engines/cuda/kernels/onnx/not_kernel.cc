#include "ppl/nn/engines/cuda/kernels/onnx/not_kernel.h"

#include "cudakernel/unary/not.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode NotKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDANotForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr<bool>(),
                                                       &output->GetShape(), output->GetBufferPtr<bool>());
    return status;
}

}}} // namespace ppl::nn::cuda
