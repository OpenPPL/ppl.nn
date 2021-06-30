#include "ppl/nn/engines/x86/optimizer/ops/onnx/not_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/not_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode NotOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* NotOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<NotKernel>();
}

}}} // namespace ppl::nn::x86
