#include "ppl/nn/engines/cuda/optimizer/ops/onnx/softmax_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/softmax_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode SoftmaxOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<SoftmaxParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = GenericInferDims;

    return RC_SUCCESS;
}

RetCode SoftmaxOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* SoftmaxOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SoftmaxKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
