#include "ppl/nn/engines/cuda/optimizer/ops/onnx/max_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/max_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_max.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode MaxOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMax(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode MaxOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* MaxOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<MaxKernel>();
}

}}} // namespace ppl::nn::cuda
