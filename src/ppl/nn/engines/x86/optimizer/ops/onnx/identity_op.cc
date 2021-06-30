#include "ppl/nn/engines/x86/optimizer/ops/onnx/identity_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/identity_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode IdentityOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* IdentityOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<IdentityKernel>();
}

}}} // namespace ppl::nn::x86
