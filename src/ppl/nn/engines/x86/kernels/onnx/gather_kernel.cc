#include "ppl/nn/engines/x86/kernels/onnx/gather_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/gather.h"
#include "ppl/kernel/x86/int64/gather.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode GatherKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const uint32_t q = indices->GetShape().GetRealDimCount();

    int64_t num_indices = 1;
    int64_t indices_dim = indices->GetShape().GetDim(q - 1);
    int64_t outter_dim = 1;
    int64_t inner_dim = 1;
    int64_t n = indices->GetShape().GetElementsExcludingPadding();
    std::vector<int64_t> real_indices;
    real_indices.resize(n);
    if (q != 0) {
        for (uint32_t i = 0; i < q - 1; ++i) {
            num_indices *= indices->GetShape().GetDim(i);
        }
        for (uint32_t i = 0; i < indices->GetShape().GetElementsExcludingPadding(); ++i) {
            real_indices[i] = indices->GetBufferPtr<int64_t>()[i] >= 0 ? indices->GetBufferPtr<int64_t>()[i]
                                                                       : indices->GetBufferPtr<int64_t>()[i] + q;
        }
    }
    if (indices->GetShape().IsScalar()) {
        real_indices[0] = indices->GetBufferPtr<int64_t>()[0] >= 0
            ? indices->GetBufferPtr<int64_t>()[0]
            : indices->GetBufferPtr<int64_t>()[0] + x->GetShape().GetDim(param_->axis);
    }
    for (int32_t i = 0; i < param_->axis; ++i) {
        outter_dim *= x->GetShape().GetDim(i);
    }
    int32_t gather_dim = x->GetShape().GetDim(param_->axis);

    for (uint32_t i = param_->axis + 1; i < x->GetShape().GetDimCount(); ++i) {
        inner_dim *= x->GetShape().GetDim(i);
    }

    const ppl::common::datatype_t data_type = y->GetShape().GetDataType();
    const auto data_format = x->GetShape().GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            return kernel::x86::gather_ndarray_fp32(x->GetBufferPtr<float>(), real_indices.data(), outter_dim,
                                                    gather_dim, inner_dim, num_indices, indices_dim,
                                                    y->GetBufferPtr<float>());
        } else if (data_type == ppl::common::DATATYPE_INT64) {
            return kernel::x86::gather_ndarray_int64(x->GetBufferPtr<int64_t>(), real_indices.data(), outter_dim,
                                                     gather_dim, inner_dim, num_indices, indices_dim,
                                                     y->GetBufferPtr<int64_t>());
        } else {
            LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type);
        }
    } else {
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
