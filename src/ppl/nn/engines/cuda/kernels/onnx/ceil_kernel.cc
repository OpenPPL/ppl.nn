#include "ppl/nn/engines/cuda/kernels/onnx/ceil_kernel.h"

#include "cudakernel/unary/unary.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode CeilKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDAUnaryCeilForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                             &output->GetShape(), output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
