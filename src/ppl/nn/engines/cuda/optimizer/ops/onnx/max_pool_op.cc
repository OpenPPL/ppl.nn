#include "ppl/nn/engines/cuda/optimizer/ops/onnx/max_pool_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/pooling_max_normal_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pooling.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode MaxPoolOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<PoolingParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        if (type == ppl::common::DATATYPE_UNKNOWN) {
            type = ppl::common::DATATYPE_FLOAT16;
        }
        auto status = InferDefaultType(info, type);
        if (info->GetOutputCount() > 1) {
            auto shape = &info->GetOutput<TensorImpl>(1)->GetShape();
            shape->SetDataType(ppl::common::DATATYPE_INT64);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapePooling(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode MaxPoolOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* MaxPoolOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PoolingMaxNormalKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
