#include "ppl/nn/engines/cuda/kernels/onnx/topk_kernel.h"

#include "cudakernel/nn/topk.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t TopKKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto indices_shape = ctx.GetOutput<TensorImpl>(1)->GetShape();
    int64_t k_value;
    auto status = ctx.GetInput<TensorImpl>(1)->CopyToHost(&k_value);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Copy k value failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    return PPLTopKGetTempBufferSize(&indices_shape, k_value, param_->axis, param_->sorted);
}

ppl::common::RetCode TopKKernel::DoExecute(KernelExecContext* ctx) {
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

    auto x = ctx->GetInput<TensorImpl>(0);
    int64_t k_value;
    auto k = ctx->GetInput<TensorImpl>(1);
    auto values = ctx->GetOutput<TensorImpl>(0);
    auto indices = ctx->GetOutput<TensorImpl>(1);
    status = k->CopyToHost(&k_value);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Copy k value failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }

    uint32_t axis = param_->axis < 0 ? param_->axis + x->GetShape().GetDimCount() : param_->axis;
    const int64_t axis_dim = x->GetShape().GetDim(axis);
    k_value = std::min(k_value, axis_dim);
    // LOG(INFO) << k_value << " -> " << axis_dim;
    status = PPLCUDATopKForwardImp(GetStream(), &x->GetShape(), x->GetBufferPtr(), &values->GetShape(),
                                   values->GetBufferPtr(), &indices->GetShape(), (int32_t*)indices->GetBufferPtr(),
                                   tmp_buffer, tmp_buffer_bytes, k_value, axis, param_->largest, param_->sorted);

    return status;
}

}}} // namespace ppl::nn::cuda
