#include "ppl/nn/engines/x86/optimizer/ops/onnx/and_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/and_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_and.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode AndOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeAnd(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_BOOL);
    };

    return RC_SUCCESS;
}

KernelImpl* AndOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<AndKernel>();
}

}}} // namespace ppl::nn::x86
