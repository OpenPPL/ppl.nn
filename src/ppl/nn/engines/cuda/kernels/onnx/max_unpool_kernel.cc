#include "ppl/nn/engines/cuda/kernels/onnx/max_unpool_kernel.h"

#include "cudakernel/nn/unpooling.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MaxUnpoolKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    // onnx version use bottom mask
    bool use_bottom_mask = true;
    int zero_int = 0;
    ppl::common::RetCode status =
        PPLCUDAMaxUnpoolForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                   output->GetBufferPtr(), use_bottom_mask, indices->GetBufferPtr<int64_t>(), zero_int,
                                   zero_int, zero_int, zero_int, zero_int, zero_int, zero_int);
    return status;
}

}}} // namespace ppl::nn::cuda
