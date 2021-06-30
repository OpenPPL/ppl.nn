#include "ppl/nn/engines/cuda/kernels/onnx/pooling_max_normal_kernel.h"

#include "cudakernel/nn/global_pooling_max.h"
#include "cudakernel/nn/pooling_max.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode PoolingMaxNormalKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_UNSUPPORTED;
    if (param_->global_pooling) {
        status = PPLCUDAGlobalMaxPoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                   &output->GetShape(), output->GetBufferPtr());
    } else {
        int32_t kernel_h = param_->kernel_shape[0];
        int32_t kernel_w = param_->kernel_shape[1];
        int32_t stride_h = param_->strides[0];
        int32_t stride_w = param_->strides[1];
        int32_t pad_h = param_->pads.size() >= 1 ? param_->pads[0] : 0;
        int32_t pad_w = param_->pads.size() >= 2 ? param_->pads[1] : 0;
        // 1*1 pooling, just transfer
        if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
            ctx->GetOutputCount() == 1 && kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 &&
            pad_h == 0 && pad_w == 0) {
            output->TransferBufferFrom(input);
            return ppl::common::RC_SUCCESS;
        }

        if (ctx->GetOutputCount() == 1) {
            status = PPLCUDAMaxPoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                 &output->GetShape(), output->GetBufferPtr(), kernel_h, kernel_w,
                                                 stride_h, stride_w, pad_h, pad_w);
        } else if (ctx->GetOutputCount() == 2) {
            auto indices = ctx->GetOutput<TensorImpl>(1);
            status = PPLCUDAMaxPoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                 &output->GetShape(), output->GetBufferPtr(), &indices->GetShape(),
                                                 indices->GetBufferPtr<int64_t>(), kernel_h, kernel_w, stride_h,
                                                 stride_w, pad_h, pad_w);
        }
    }

    return status;
}

}}} // namespace ppl::nn::cuda
