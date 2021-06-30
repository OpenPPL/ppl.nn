#include "ppl/nn/engines/cuda/kernels/onnx/shape_kernel.h"

#include "cudakernel/arithmetic/arithmetic.h"

namespace ppl { namespace nn { namespace cuda {

bool ShapeKernel::CanDoExecute(const KernelExecContext&) const {
    return true;
}

ppl::common::RetCode ShapeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto shape = ctx->GetOutput<TensorImpl>(0);

    std::unique_ptr<int64_t[]> shape_host(new int64_t[shape->GetShape().GetElementsIncludingPadding()]);
    for (size_t i = 0; i < data->GetShape().GetRealDimCount(); i++) {
        shape_host[i] = data->GetShape().GetDim(i);
    }
    cudaMemcpyAsync(shape->GetBufferPtr(), shape_host.get(), shape->GetShape().GetBytesIncludingPadding(),
                    cudaMemcpyHostToDevice, GetStream());

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
