#include "ppl/nn/engines/x86/kernels/onnx/roialign_kernel.h"

#include "ppl/kernel/x86/fp32/roialign.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ROIAlignKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto rois = ctx->GetInput<TensorImpl>(1);
    auto batch_indices = ctx->GetInput<TensorImpl>(2);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [rois]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(rois);
    PPLNN_X86_DEBUG_TRACE("Input [batch_indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(batch_indices);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    auto data_format = x->GetShape().GetDataFormat();
    auto data_type = x->GetShape().GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (data_format == ppl::common::DATAFORMAT_N16CX) {
            if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return kernel::x86::roialign_n16cx_fp32_avx512(
                    &x->GetShape(), &rois->GetShape(), &batch_indices->GetShape(), x->GetBufferPtr<float>(),
                    rois->GetBufferPtr<float>(), batch_indices->GetBufferPtr<int64_t>(), param_->mode,
                    param_->output_height, param_->output_width, param_->sampling_ratio, param_->spatial_scale,
                    y->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return kernel::x86::roialign_n16cx_fp32_avx(
                    &x->GetShape(), &rois->GetShape(), &batch_indices->GetShape(), x->GetBufferPtr<float>(),
                    rois->GetBufferPtr<float>(), batch_indices->GetBufferPtr<int64_t>(), param_->mode,
                    param_->output_height, param_->output_width, param_->sampling_ratio, param_->spatial_scale,
                    y->GetBufferPtr<float>());
            } else {
                return kernel::x86::roialign_n16cx_fp32(
                    &x->GetShape(), &rois->GetShape(), &batch_indices->GetShape(), x->GetBufferPtr<float>(),
                    rois->GetBufferPtr<float>(), batch_indices->GetBufferPtr<int64_t>(), param_->mode,
                    param_->output_height, param_->output_width, param_->sampling_ratio, param_->spatial_scale,
                    y->GetBufferPtr<float>());
            }
        } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::roialign_ndarray_fp32(
                &x->GetShape(), &rois->GetShape(), &batch_indices->GetShape(), x->GetBufferPtr<float>(),
                rois->GetBufferPtr<float>(), batch_indices->GetBufferPtr<int64_t>(), param_->mode,
                param_->output_height, param_->output_width, param_->sampling_ratio, param_->spatial_scale,
                y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else {
        LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
