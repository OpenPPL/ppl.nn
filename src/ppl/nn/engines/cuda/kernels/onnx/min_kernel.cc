#include "ppl/nn/engines/cuda/kernels/onnx/min_kernel.h"

#include "cudakernel/arithmetic/arithmetic.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MinKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status =
        PPLCUDAArithMeticMinForwardImp(GetStream(), &input0->GetShape(), input0->GetBufferPtr(), &input1->GetShape(),
                                       input1->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr());

    int32_t input_count = ctx->GetInputCount();
    if (input_count > 2) {
        for (int it = 2; it < input_count; ++it) {
            auto input = ctx->GetInput<TensorImpl>(it);
            status = PPLCUDAArithMeticMinForwardImp(GetStream(), &output->GetShape(), output->GetBufferPtr(),
                                                    &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                                    output->GetBufferPtr());
        }
    }
    return status;
}

}}} // namespace ppl::nn::cuda
