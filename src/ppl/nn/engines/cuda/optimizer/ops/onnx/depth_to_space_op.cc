#include "ppl/nn/engines/cuda/optimizer/ops/onnx/depth_to_space_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/depth_to_space_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_depth_to_space.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode DepthToSpaceOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<DepthToSpaceParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeDepthToSpace(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode DepthToSpaceOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* DepthToSpaceOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<DepthToSpaceKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
