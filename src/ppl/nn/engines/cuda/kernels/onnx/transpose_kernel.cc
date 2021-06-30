#include "ppl/nn/engines/cuda/kernels/onnx/transpose_kernel.h"

#include "cudakernel/memory/transpose.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode TransposeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    const TensorShape& in_shape0 = input->GetShape();

    ppl::nn::common::TransposeParam modified_param = *param_;
    if (modified_param.perm.empty()) {
        int32_t dim_count = in_shape0.GetDimCount();
        modified_param.perm.resize(dim_count);
        for (int it = 0; it < dim_count; ++it) {
            modified_param.perm[it] = dim_count - it - 1;
        }
    }

    ppl::common::RetCode status =
        PPLCUDATransposeForwardImp(GetStream(), modified_param, &input->GetShape(), input->GetBufferPtr(),
                                   &output->GetShape(), output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
