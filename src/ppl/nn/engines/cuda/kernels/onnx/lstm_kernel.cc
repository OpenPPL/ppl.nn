#include "ppl/nn/engines/cuda/kernels/onnx/lstm_kernel.h"
#include "cudakernel/nn/lstm.h"

namespace ppl { namespace nn { namespace cuda {

bool LstmKernel::CanDoExecute(const KernelExecContext& ctx) const {
    if( !(
          ctx.GetInput<TensorImpl>(0) &&
          ctx.GetInput<TensorImpl>(1) &&
          ctx.GetInput<TensorImpl>(2) &&
          ctx.GetInput<TensorImpl>(3)
         )
      )
        return false;
    if( !(
          ctx.GetOutput<TensorImpl>(0) ||
          ctx.GetOutput<TensorImpl>(1) ||
          ctx.GetOutput<TensorImpl>(2)
         )
      )
        return false;

    return true;
}


ppl::common::RetCode LstmKernel::DoExecute(KernelExecContext* ctx) {
    
    auto X = ctx->GetInput<TensorImpl>(0);
    auto W = ctx->GetInput<TensorImpl>(1);
    auto R = ctx->GetInput<TensorImpl>(2);
    auto B = ctx->GetInput<TensorImpl>(3);
    auto sequence_lens = ctx->GetInput<TensorImpl>(4);

    auto initial_h = ctx->GetInput<TensorImpl>(5);
    auto initial_c = ctx->GetInput<TensorImpl>(6);
    auto P = ctx->GetInputCount() >= 8 ? ctx->GetInput<TensorImpl>(7) : NULL;
    auto Y = ctx->GetOutput<TensorImpl>(0);
    auto Y_h = ctx->GetOutput<TensorImpl>(1);
    auto Y_c = ctx->GetOutput<TensorImpl>(2);

    auto X_shape = X->GetShape();
    auto hidden_size = param_->hidden_size;
    int64_t size = PPLCUDALstmGetRuntimeBufSize(&X_shape, direction_, hidden_size);
    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(size, &tmp_buffer_desc);
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto seq_lens_ptr = sequence_lens? sequence_lens->GetBufferPtr() : NULL;
    auto initial_h_ptr = initial_h? initial_h->GetBufferPtr() : NULL;
    auto initial_c_ptr = initial_c? initial_c->GetBufferPtr() : NULL;
    auto Y_ptr = Y? Y->GetBufferPtr() : NULL;
    auto Y_h_ptr = Y_h? Y_h->GetBufferPtr() : NULL;
    auto Y_c_ptr = Y_c? Y_c->GetBufferPtr() : NULL;
    auto P_ptr = P ? P->GetBufferPtr() : NULL;

    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    status = PPLCUDALstmForwardImp(GetStream(), module, &X_shape, X->GetBufferPtr(),
                              W->GetBufferPtr(), R->GetBufferPtr(), P_ptr, B->GetBufferPtr(),
                              seq_lens_ptr, initial_h_ptr, initial_c_ptr,
                              direction_, hidden_size, tmp_buffer,
                              Y_ptr, Y_h_ptr, Y_c_ptr);
    return status;
}

}}} // namespace ppl::nn::cuda
