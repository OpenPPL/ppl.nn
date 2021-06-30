#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sigmoid_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/sigmoid_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode SigmoidOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode SigmoidOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* SigmoidOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<SigmoidKernel>();
}

}}} // namespace ppl::nn::cuda
