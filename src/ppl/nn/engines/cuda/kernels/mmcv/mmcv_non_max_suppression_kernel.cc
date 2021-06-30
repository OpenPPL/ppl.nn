#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_non_max_suppression_kernel.h"

#include "cudakernel/nn/mmcv_nms.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t MMCVNonMaxSuppressionKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto boxes = ctx.GetInput<TensorImpl>(0);
    return PPLMMCVNMSGetTempBufferSize(&boxes->GetShape());
}

ppl::common::RetCode MMCVNonMaxSuppressionKernel::DoExecute(KernelExecContext* ctx) {
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

    auto boxes = ctx->GetInput<TensorImpl>(0);
    auto scores = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    int device_id = GetCudaDevice()->GetDeviceId();
    status = PPLCUDAMMCVNMSForwardImp(GetStream(), &boxes->GetShape(), boxes->GetBufferPtr(), &scores->GetShape(),
                                      scores->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr<int64_t>(),
                                      tmp_buffer, tmp_buffer_bytes, device_id, param_->iou_threshold, param_->offset);

    return status;
}

}}} // namespace ppl::nn::cuda
