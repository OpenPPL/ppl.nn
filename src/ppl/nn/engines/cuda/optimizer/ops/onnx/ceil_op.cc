#include "ppl/nn/engines/cuda/optimizer/ops/onnx/ceil_op.h"

#include "ppl/nn/engines/cuda/kernels/onnx/ceil_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_ceil.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CeilOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(DATATYPE_FLOAT32);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeCeil(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode CeilOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* CeilOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<CeilKernel>();
}

}}} // namespace ppl::nn::cuda
