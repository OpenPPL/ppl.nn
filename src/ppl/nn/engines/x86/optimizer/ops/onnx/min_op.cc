#include "ppl/nn/engines/x86/optimizer/ops/onnx/min_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/min_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_min.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MinOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMin(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* MinOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<MinKernel>();
}

}}} // namespace ppl::nn::x86
