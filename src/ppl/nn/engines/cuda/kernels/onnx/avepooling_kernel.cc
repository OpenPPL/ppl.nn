#include "ppl/nn/engines/cuda/kernels/onnx/avepooling_kernel.h"

#include "cudakernel/nn/pooling_ave.h"
#include "cudakernel/nn/global_pooling_ave.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode AvePoolingKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_UNSUPPORTED;
    if (param_->global_pooling) {
        status = PPLCUDAGlobalAvePoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                   &output->GetShape(), output->GetBufferPtr());
    } else {
        int32_t kernel_h = param_->kernel_shape[0];
        int32_t kernel_w = param_->kernel_shape[1];
        int32_t stride_h = param_->strides[0];
        int32_t stride_w = param_->strides[1];
        int32_t pad_h = param_->pads.size() >= 1 ? param_->pads[0] : 0;
        int32_t pad_w = param_->pads.size() >= 2 ? param_->pads[1] : 0;
        // 1*1 pooling, just transfer
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
            output->TransferBufferFrom(input);
            return ppl::common::RC_SUCCESS;
        }

        int32_t if_excluding_padding = 1;
        if (param_->mode == ppl::nn::common::PoolingParam::POOLING_AVERAGE_INCLUDE) {
            if_excluding_padding = 0;
        }
        status = PPLCUDAAvePoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                             &output->GetShape(), output->GetBufferPtr(), kernel_h, kernel_w, stride_h,
                                             stride_w, pad_h, pad_w, if_excluding_padding);
    }
    return status;
}

}}} // namespace ppl::nn::cuda
