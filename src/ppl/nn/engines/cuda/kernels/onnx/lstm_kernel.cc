#include "ppl/nn/engines/cuda/kernels/onnx/lstm_kernel.h"
#include "cudakernel/nn/lstm.h"

namespace ppl { namespace nn { namespace cuda {
ppl::common::RetCode LstmKernel::DoExecute(KernelExecContext* ctx) {
    
    auto X = ctx->GetInput<TensorImpl>(0);
    auto W = ctx->GetInput<TensorImpl>(1);
    auto R = ctx->GetInput<TensorImpl>(2);
    auto B = ctx->GetInput<TensorImpl>(3);
    auto sequence_lens = ctx->GetInput<TensorImpl>(4);
    auto initial_h = ctx->GetInput<TensorImpl>(5);
    auto initial_c = ctx->GetInput<TensorImpl>(6);
    auto P = ctx->GetInput<TensorImpl>(7);
    auto Y = ctx->GetOutput<TensorImpl>(0);
    auto Y_h = ctx->GetOutput<TensorImpl>(1);
    auto Y_c = ctx->GetOutput<TensorImpl>(2);

    auto X_shape = X->GetShape();
    auto hidden_size = param_->hidden_size;
    int64_t size = PPLCUDALstmGetRuntimeBufSize(&X_shape, direction_, hidden_size);
    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(size, &tmp_buffer_desc);
    auto tmp_buffer = tmp_buffer_desc.addr;

    status = PPLCUDALstmForwardImp(GetStream(), &X_shape, X->GetBufferPtr(),
                              W->GetBufferPtr(), R->GetBufferPtr(), P->GetBufferPtr(), B->GetBufferPtr(),
                              sequence_lens->GetBufferPtr(), initial_h->GetBufferPtr(), initial_c->GetBufferPtr(),
                              direction_, hidden_size, tmp_buffer,
                              Y->GetBufferPtr(), Y_h->GetBufferPtr(), Y_c->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
