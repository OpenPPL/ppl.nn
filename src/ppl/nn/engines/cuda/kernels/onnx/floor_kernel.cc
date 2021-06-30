#include "ppl/nn/engines/cuda/kernels/onnx/floor_kernel.h"

#include "cudakernel/unary/unary.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode FloorKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDAUnaryFloorForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                              &output->GetShape(), output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
