#include "ppl/nn/engines/cuda/kernels/onnx/split_kernel.h"

#include "cudakernel/memory/split.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode SplitKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    int32_t dim_count = input->GetShape().GetDimCount();
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        ctx->GetOutputCount() == 1 &&
        input->GetShape().GetElementsIncludingPadding() ==
            ctx->GetOutput<TensorImpl>(0)->GetShape().GetElementsIncludingPadding()) {
        auto output = ctx->GetOutput<TensorImpl>(0);
        output->TransferBufferFrom(input);
        return ppl::common::RC_SUCCESS;
    }

    dst_dims_.resize(ctx->GetOutputCount());
    dst_list_.resize(ctx->GetOutputCount());

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto output = ctx->GetOutput<TensorImpl>(i);
        dst_list_[i] = output->GetBufferPtr();
        for (int32_t it = 0; it < dim_count; ++it) {
            dst_dims_[i].push_back(output->GetShape().GetDim(it));
        }
    }

    typedef int64_t* pint;
    std::vector<pint> out_dims(ctx->GetOutputCount());
    for (uint32_t it = 0; it < ctx->GetOutputCount(); ++it) {
        out_dims[it] = dst_dims_[it].data();
    }

    ppl::common::RetCode status = PPLCUDASplitForwardImp(
        GetStream(), (param_->axis + dim_count) % dim_count, &input->GetShape(), input->GetBufferPtr(),
        ctx->GetOutputCount(), (const int64_t**)out_dims.data(), dst_list_.data());
    return status;
}

}}} // namespace ppl::nn::cuda
