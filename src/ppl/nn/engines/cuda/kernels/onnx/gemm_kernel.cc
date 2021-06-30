#include "ppl/nn/engines/cuda/kernels/onnx/gemm_kernel.h"

#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/gemm.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t GemmKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto A = &ctx.GetInput<TensorImpl>(0)->GetShape();
    return PPLGemmCUDAGetBufSize(A, param_->param.transA);
}

ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
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
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    TensorShape bias_shape;
    void* bias = nullptr;
    if (ctx->GetInputCount() >= 3) {
        bias_shape = ctx->GetInput<TensorImpl>(2)->GetShape();
        bias = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
    }

    fuse_param_t temp_fuse_param;
    ConvertToEmptyFuseParam(temp_fuse_param);
    temp_fuse_param.has_activation = param_->extra_param.has_activation;
    temp_fuse_param.has_clip = param_->extra_param.has_clip;
    temp_fuse_param.clip_min = param_->extra_param.clip.min_val;
    temp_fuse_param.clip_max = param_->extra_param.clip.max_val;

    auto stream = GetStream();
    status = PPLCUDAGemmForwardImp(stream, &input->GetShape(), input->GetBufferPtr(), &weight->GetShape(),
                                   weight->GetBufferPtr(), bias, &output->GetShape(), output->GetBufferPtr(),
                                   param_->param, tmp_buffer, temp_fuse_param, param_->extra_param.kernel_index);

    return status;
}

}}} // namespace ppl::nn::cuda
