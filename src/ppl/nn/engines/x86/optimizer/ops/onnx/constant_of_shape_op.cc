#include "ppl/nn/engines/x86/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_constant_of_shape.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ConstantOfShapeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeConstantOfShape(info, param_.get());
    };

    infer_type_func_ = [this](InputOutputInfo* info) -> void {
        TensorShape* shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(param_->data_type);
    };

    return RC_SUCCESS;
}

KernelImpl* ConstantOfShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConstantOfShapeKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
