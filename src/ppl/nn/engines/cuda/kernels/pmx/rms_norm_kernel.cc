#include "ppl/nn/engines/cuda/kernels/pmx/rms_norm_kernel.h"
#include "cudakernel/nn/rms_norm.h"

namespace ppl { namespace nn { namespace cuda {


ppl::common::RetCode RMSNormKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto in_shape = input->GetShape();
    auto weight = ctx->GetInput<TensorImpl>(1);
    void* skip = nullptr;
    if (ctx->GetInputCount() == 3) {
        skip = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
    }

    if (param_->skip_term == false) {
        LOG(ERROR) << "only support SkipRMSNorm now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    auto output1 = ctx->GetOutput<TensorImpl>(0);
    auto output2 = ctx->GetOutput<TensorImpl>(1);

    LOG(DEBUG) << "Run RMSNormKernel with datatype " << in_shape->GetDataType() << " dataformat " << in_shape->GetDataFormat();

    auto status = PPLCUDARmsNormForwardImp(GetStream(), input->GetBufferPtr(), skip, weight->GetBufferPtr(), param_->eps, in_shape,
                                            output1->GetBufferPtr(), output2->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
