#include "ppl/nn/engines/cuda/optimizer/ops/onnx/not_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/not_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode NotOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto in_shape = &info->GetInput<TensorImpl>(0)->GetShape();
        in_shape->SetDataType(DATATYPE_BOOL);
        auto out_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        out_shape->SetDataType(DATATYPE_BOOL);
        return RC_SUCCESS;
    };

    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode NotOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* NotOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<NotKernel>();
}

}}} // namespace ppl::nn::cuda
