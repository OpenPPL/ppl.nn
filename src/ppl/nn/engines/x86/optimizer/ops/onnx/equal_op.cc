#include "ppl/nn/engines/x86/optimizer/ops/onnx/equal_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/equal_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_equal.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode EqualOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeEqual(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_BOOL);
    };

    return RC_SUCCESS;
}

KernelImpl* EqualOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<EqualKernel>();
}

}}} // namespace ppl::nn::x86
