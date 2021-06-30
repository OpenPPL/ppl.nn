#include "ppl/nn/engines/x86/optimizer/ops/onnx/scatter_elements_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/scatter_elements_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_scatter_elements.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ScatterElementsOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeScatterElements(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* ScatterElementsOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ScatterElementsKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
