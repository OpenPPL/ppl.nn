#include "ppl/nn/engines/x86/optimizer/ops/onnx/less_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/less_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_less.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode LessOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        if (info->GetInput<TensorImpl>(0)->GetShape().GetDataFormat() != DATAFORMAT_NDARRAY) {
            return RC_UNSUPPORTED;
        }
        return oputils::ReshapeLess(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_BOOL);
    };

    return RC_SUCCESS;
}

KernelImpl* LessOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<LessKernel>();
}

}}} // namespace ppl::nn::x86
