#include "ppl/nn/engines/cuda/kernels/pmx/layernorm_kernel.h"
#include "cudakernel/nn/layernorm.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode LayerNormKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto in_shape0 = input->GetShape();
    auto output = ctx->GetOutput<TensorImpl>(0);
    void* scale_ptr = nullptr;
    void* shift_ptr = nullptr;
    if (param_->elementwise_affine) {
        scale_ptr = ctx->GetInput<TensorImpl>(1)->GetBufferPtr();
        shift_ptr = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
    }

    int axis = param_->axis;
    int64_t outer = 1;
    int64_t inner = 1;

    if (axis < 0) {
        axis += in_shape0->GetDimCount();
    }

    for (int32_t i = 0; i < axis; ++i) {
        outer *= in_shape0->GetDim(i);
    }
    for (uint32_t i = axis; i < in_shape0->GetDimCount(); ++i) {
        inner *= in_shape0->GetDim(i);
    }

    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);

    LOG(DEBUG) << "Run LayerNormKernel with datatype " << in_shape0->GetDataType() << " dataformat "
               << in_shape0->GetDataFormat();

    auto status = PPLCUDALayerNormForwardImp(GetStream(), in_shape0, input->GetBufferPtr(), scale_ptr, shift_ptr,
                                             output->GetBufferPtr(), outer, inner, param_->elementwise_affine,
                                             param_->eps, input_quant.scale[0], 1.0f / output_quant.scale[0]);
    return status;
}

}}} // namespace ppl::nn::cuda
