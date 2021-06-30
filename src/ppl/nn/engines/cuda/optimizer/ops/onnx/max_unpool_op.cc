#include "ppl/nn/engines/cuda/optimizer/ops/onnx/max_unpool_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/max_unpool_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_maxunpool.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode MaxUnPoolOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<MaxUnpoolParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        if (type == ppl::common::DATATYPE_UNKNOWN)
            type = ppl::common::DATATYPE_FLOAT16;
        auto status = InferDefaultType(info, type);
        auto shape1 = &info->GetInput<TensorImpl>(1)->GetShape();
        shape1->SetDataType(DATATYPE_INT64);
        if (info->GetInputCount() > 2) {
            auto shape2 = &info->GetInput<TensorImpl>(2)->GetShape();
            shape2->SetDataType(DATATYPE_INT64);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMaxUnpool(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode MaxUnPoolOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* MaxUnPoolOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MaxUnpoolKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
