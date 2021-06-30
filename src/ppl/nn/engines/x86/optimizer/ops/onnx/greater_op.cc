#include "ppl/nn/engines/x86/optimizer/ops/onnx/greater_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/greater_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_greater.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode GreaterOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeGreater(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_BOOL);
    };

    return RC_SUCCESS;
}

KernelImpl* GreaterOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<GreaterKernel>();
}

}}} // namespace ppl::nn::x86
