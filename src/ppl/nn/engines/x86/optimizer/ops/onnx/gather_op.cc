#include "ppl/nn/engines/x86/optimizer/ops/onnx/gather_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/gather_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_gather.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode GatherOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeGather(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* GatherOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<GatherKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
