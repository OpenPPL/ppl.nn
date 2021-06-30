#include "ppl/nn/engines/x86/optimizer/ops/onnx/cast_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/cast_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_cast.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode CastOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeCast(info, param_.get());
    };

    infer_type_func_ = [this](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(this->param_->to);
    };

    return RC_SUCCESS;
}

KernelImpl* CastOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<CastKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
