#include "ppl/nn/engines/x86/optimizer/ops/onnx/non_zero_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/non_zero_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_non_zero.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode NonZeroOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeNonZero(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

KernelImpl* NonZeroOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<NonZeroKernel>();
}

}}} // namespace ppl::nn::x86
