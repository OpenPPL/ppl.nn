#include "ppl/nn/engines/cuda/kernels/bridge_kernel.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace cuda {

bool BridgeKernel::EqualTypeAndFormat(const TensorImpl* input, const TensorImpl* output) {
    if (input->GetShape().GetDataType() != output->GetShape().GetDataType()) {
        return false;
    }

    if (input->GetShape().GetDataFormat() == output->GetShape().GetDataFormat()) {
        return true;
    }

    if (input->GetShape().GetDimCount() == 2 && output->GetShape().GetDimCount() == 2) {
        return true;
    }

    return false;
}

ppl::common::RetCode BridgeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    auto converter = output->GetDevice()->GetDataConverter();
    status =
        converter->Convert(&output->GetBufferDesc(), output->GetShape(), input->GetBufferDesc(), input->GetShape());
    return status;
}

}}} // namespace ppl::nn::cuda
