#include "ppl/nn/engines/x86/optimizer/ops/onnx/gather_nd_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/gather_nd_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_gather_nd.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode GatherNDOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeGatherND(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* GatherNDOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<GatherNdKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
