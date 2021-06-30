#include "ppl/nn/engines/x86/optimizer/ops/onnx/squeeze_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/squeeze_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_squeeze.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SqueezeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeSqueeze(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* SqueezeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SqueezeKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
