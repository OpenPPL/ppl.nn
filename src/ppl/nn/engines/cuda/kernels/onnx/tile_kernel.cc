#include "ppl/nn/engines/cuda/kernels/onnx/tile_kernel.h"

#include <memory>

#include "cudakernel/memory/tile.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode TileKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    TileParam kernel_param;

    if (ctx->GetInputCount() >= 2) {
        auto constant_data = ctx->GetInput<TensorImpl>(1);
        auto status = constant_data->CopyToHost(&(kernel_param.repeats));
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy repeats failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    ppl::common::RetCode status =
        PPLCUDATileForwardImp(GetStream(), kernel_param, &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                              output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
