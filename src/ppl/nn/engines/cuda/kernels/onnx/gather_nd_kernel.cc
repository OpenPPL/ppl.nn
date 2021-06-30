#include "ppl/nn/engines/cuda/kernels/onnx/gather_nd_kernel.h"

#include "cudakernel/memory/gather_nd.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t GatherNdKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto input = ctx.GetInput<TensorImpl>(0);
    auto indices = ctx.GetInput<TensorImpl>(1);
    return pplGatherNDGetTempBufferSize(&input->GetShape(), input->GetBufferPtr(), &indices->GetShape(),
                                        indices->GetBufferPtr());
}

ppl::common::RetCode GatherNdKernel::DoExecute(KernelExecContext* ctx) {
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
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    status =
        PPLCUDAGatherNDForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &indices->GetShape(),
                                  indices->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), tmp_buffer);
    return status;
}

}}} // namespace ppl::nn::cuda
