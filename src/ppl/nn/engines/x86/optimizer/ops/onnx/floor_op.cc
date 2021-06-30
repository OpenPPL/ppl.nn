#include "ppl/nn/engines/x86/optimizer/ops/onnx/floor_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/floor_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode FloorOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* FloorOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<FloorKernel>();
}

}}} // namespace ppl::nn::x86
