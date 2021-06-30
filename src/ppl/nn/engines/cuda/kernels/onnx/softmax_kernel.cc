#include "ppl/nn/engines/cuda/kernels/onnx/softmax_kernel.h"

#include "cudakernel/nn/softmax.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t SoftmaxKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto input = ctx.GetInput<TensorImpl>(0);
    return PPLSoftmaxGetTempBufferSize(&input->GetShape(), param_->axis);
}

ppl::common::RetCode SoftmaxKernel::DoExecute(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    status = PPLCUDASoftmaxForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                      output->GetBufferPtr(), tmp_buffer, param_->axis);
    return status;
}

}}} // namespace ppl::nn::cuda
