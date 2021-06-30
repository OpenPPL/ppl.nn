#include "ppl/nn/engines/x86/optimizer/ops/onnx/expand_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/expand_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_expand.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ExpandOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeExpand(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* ExpandOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ExpandKernel>();
}

}}} // namespace ppl::nn::x86
