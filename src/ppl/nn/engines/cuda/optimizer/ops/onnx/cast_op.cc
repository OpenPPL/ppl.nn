#include "ppl/nn/engines/cuda/optimizer/ops/onnx/cast_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/cast_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_cast.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CastOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<CastParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto output = &info->GetOutput<TensorImpl>(0)->GetShape();
        output->SetDataType(param_.to);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeCast(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode CastOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* CastOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<CastKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
