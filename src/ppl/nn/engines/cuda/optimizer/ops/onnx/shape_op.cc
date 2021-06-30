#include "ppl/nn/engines/cuda/optimizer/ops/onnx/shape_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/shape_kernel.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ShapeOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(ppl::common::DATATYPE_INT64);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto output_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->Reshape({info->GetInput<TensorImpl>(0)->GetShape().GetRealDimCount()});

        return RC_SUCCESS;
    };

    return RC_SUCCESS;
}

RetCode ShapeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ShapeKernel>();
}

}}} // namespace ppl::nn::cuda
