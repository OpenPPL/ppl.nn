#include "ppl/nn/engines/cuda/kernels/onnx/exp_kernel.h"

#include "cudakernel/unary/exp.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ExpKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDAExpForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                       &output->GetShape(), output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
