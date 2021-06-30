#include "ppl/nn/engines/cuda/optimizer/ops/onnx/gather_nd_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/gather_nd_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_gather_nd.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode GatherNDOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<GatherNDParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto status = type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
        auto shape = &info->GetInput<TensorImpl>(1)->GetShape();
        shape->SetDataType(DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeGatherND(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode GatherNDOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* GatherNDOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<GatherNdKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
