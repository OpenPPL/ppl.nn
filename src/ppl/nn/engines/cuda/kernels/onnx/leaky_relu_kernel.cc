#include "ppl/nn/engines/cuda/kernels/onnx/leaky_relu_kernel.h"

#include "cudakernel/unary/leakyrelu.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode LeakyReluKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status =
        PPLCUDAUnaryLeakyReluForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                        output->GetBufferPtr(), param_->alpha);
    return status;
}

}}} // namespace ppl::nn::cuda
