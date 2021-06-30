#include "ppl/nn/engines/x86/optimizer/ops/onnx/ceil_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/ceil_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode CeilOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* CeilOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<CeilKernel>();
}

}}} // namespace ppl::nn::x86
