#include "ppl/nn/engines/cuda/optimizer/ops/onnx/scatter_nd_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/scatter_nd_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_scatter_nd.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ScatterNDOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto status = type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
        auto shape = &info->GetInput<TensorImpl>(1)->GetShape();
        shape->SetDataType(DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeScatterND(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode ScatterNDOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ScatterNDOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ScatterNdKernel>();
}

}}} // namespace ppl::nn::cuda
