#include "ppl/nn/engines/x86/optimizer/ops/onnx/softmax_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/softmax_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SoftmaxOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* SoftmaxOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SoftmaxKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
