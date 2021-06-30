#include "ppl/nn/engines/cuda/kernels/onnx/where_kernel.h"

#include "cudakernel/memory/where.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode WhereKernel::DoExecute(KernelExecContext* ctx) {
    auto cond = ctx->GetInput<TensorImpl>(0);
    auto x = ctx->GetInput<TensorImpl>(1);
    auto y = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDAWhereForwardImp(
        GetStream(), &cond->GetShape(), (const bool*)cond->GetBufferPtr(), &x->GetShape(), x->GetBufferPtr(),
        &y->GetShape(), y->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr());

    return status;
}

}}} // namespace ppl::nn::cuda
