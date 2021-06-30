#include "ppl/nn/engines/x86/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/where_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_where.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode WhereOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeWhere(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        GenericInferType(info);
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(info->GetInput<TensorImpl>(1)->GetShape().GetDataType());
    };

    return RC_SUCCESS;
}

KernelImpl* WhereOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<WhereKernel>();
}

}}} // namespace ppl::nn::x86
