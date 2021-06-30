#include "ppl/nn/engines/x86/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/slice_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_slice.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SliceOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        // check parameters
        if (info->GetInputCount() < 3 || info->GetInputCount() > 5 || info->GetOutputCount() != 1) {
            return RC_INVALID_VALUE;
        }
        for (size_t i = 1; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape().GetDimCount() !=
                1) { // starts, ends, axes, steps must be 1-D tensor
                return RC_INVALID_VALUE;
            }
        }
        const uint32_t axes_num = info->GetInput<TensorImpl>(1)->GetShape().GetDim(0);
        for (size_t i = 2; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape().GetDim(0) !=
                axes_num) { // starts, end, axes, steps must have same length except for not defined
                return RC_INVALID_VALUE;
            }
        }
        // check support
        if (info->GetInput<TensorImpl>(0)->GetShape().GetDataType() != DATATYPE_FLOAT32 &&
            info->GetInput<TensorImpl>(0)->GetShape().GetDataType() != DATATYPE_INT64) {
            LOG(ERROR) << "unsupported data type: "
                       << GetDataTypeStr(info->GetInput<TensorImpl>(0)->GetShape().GetDataType());
            return RC_UNSUPPORTED;
        }
        for (size_t i = 1; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape().GetDataType() != DATATYPE_INT64) {
                LOG(ERROR) << "starts, ends, axes & steps only support int64 now.";
                return RC_UNSUPPORTED;
            }
        }
        for (size_t i = 0; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape().GetDataFormat() != DATAFORMAT_NDARRAY) {
                LOG(ERROR) << "unsupported data format: "
                           << GetDataFormatStr(info->GetInput<TensorImpl>(i)->GetShape().GetDataFormat());
                return RC_UNSUPPORTED;
            }
        }

        auto ret = oputils::ReshapeSlice(info);
        if (ret != RC_SUCCESS) {
            return ret;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* SliceOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<SliceKernel>();
}

}}} // namespace ppl::nn::x86
