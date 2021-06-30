#include "ppl/nn/engines/cuda/kernels/onnx/range_kernel.h"

#include "cudakernel/nn/range.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode RangeKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    return PPLCUDARangeForwardImp(GetStream(), ctx->GetInput<TensorImpl>(0)->GetBufferPtr(),
                                  ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), &output->GetShape(),
                                  output->GetBufferPtr());
}

}}} // namespace ppl::nn::cuda
