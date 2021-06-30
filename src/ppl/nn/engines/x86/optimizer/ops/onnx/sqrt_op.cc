#include "ppl/nn/engines/x86/optimizer/ops/onnx/sqrt_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/sqrt_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SqrtOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* SqrtOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<SqrtKernel>();
}

}}} // namespace ppl::nn::x86
