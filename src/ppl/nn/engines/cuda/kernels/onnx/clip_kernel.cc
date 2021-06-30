#include "ppl/nn/engines/cuda/kernels/onnx/clip_kernel.h"

#include <float.h>

#include "cudakernel/unary/clip.h"

namespace ppl { namespace nn { namespace cuda {

bool ClipKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto tensor = ctx.GetInput<TensorImpl>(0);
    if (!tensor || tensor->GetShape().GetBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ClipKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    float min_val = -FLT_MAX, max_val = FLT_MAX;
    if (ctx->GetInputCount() >= 2 && ctx->GetInput<TensorImpl>(1) != nullptr) {
        auto status = GetDevice()->CopyToHost(&min_val, ctx->GetInput<TensorImpl>(1)->GetBufferDesc(),
                                              ctx->GetInput<TensorImpl>(1)->GetShape());
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy min value failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }
    if (ctx->GetInputCount() >= 3 && ctx->GetInput<TensorImpl>(2) != nullptr) {
        auto status = GetDevice()->CopyToHost(&max_val, ctx->GetInput<TensorImpl>(2)->GetBufferDesc(),
                                              ctx->GetInput<TensorImpl>(2)->GetShape());
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy max value failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    ppl::common::RetCode status = PPLCUDAClipForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                        &output->GetShape(), output->GetBufferPtr(), min_val, max_val);
    return status;
}

}}} // namespace ppl::nn::cuda
