#include "ppl/nn/engines/cuda/kernels/onnx/and_kernel.h"

#include "cudakernel/arithmetic/logical.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode AndKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status =
        PPLCUDALogicalAndForwardImp(GetStream(), &input0->GetShape(), input0->GetBufferPtr<bool>(), &input1->GetShape(),
                                    input1->GetBufferPtr<bool>(), &output->GetShape(), output->GetBufferPtr<bool>());
    return status;
}

}}} // namespace ppl::nn::cuda
