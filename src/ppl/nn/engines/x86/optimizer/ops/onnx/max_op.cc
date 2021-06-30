#include "ppl/nn/engines/x86/optimizer/ops/onnx/max_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/max_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_max.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MaxOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMax(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* MaxOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<MaxKernel>();
}

}}} // namespace ppl::nn::x86
