#include "ppl/nn/engines/x86/optimizer/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/reshape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reshape.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ReshapeOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeReshape(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* ReshapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReshapeKernel>();
}

}}} // namespace ppl::nn::x86
