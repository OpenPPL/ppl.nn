#include "ppl/nn/engines/cuda/optimizer/ops/onnx/scatter_elements_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/scatter_elements_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_scatter_elements.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ScatterElementsOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ScatterElementsParam>(options, &param_);
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
        return oputils::ReshapeScatterElements(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode ScatterElementsOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ScatterElementsOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ScatterElementsKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
