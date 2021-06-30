#include "ppl/nn/engines/x86/optimizer/ops/onnx/log_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/log_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode LogOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* LogOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<LogKernel>();
}

}}} // namespace ppl::nn::x86
