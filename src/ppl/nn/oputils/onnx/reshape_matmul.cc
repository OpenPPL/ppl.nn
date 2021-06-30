#include "ppl/nn/oputils/onnx/reshape_matmul.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/oputils/broadcast.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeMatMul(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 2) {
        LOG(ERROR) << "2 input required.";
        return RC_INVALID_VALUE;
    }

    const TensorShape& lhs = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& rhs = info->GetInput<TensorImpl>(1)->GetShape();

    MatMulBroadCaster matmul_bc;
    matmul_bc.SetInputTensorShapes(lhs, rhs);
    if (!matmul_bc.CanBroadCast()) {
        return RC_INVALID_VALUE;
    }

    auto& output_shape = matmul_bc.OutputTensorShape();
    if (output_shape.IsScalar()) {
        info->GetOutput<TensorImpl>(0)->GetShape().ReshapeAsScalar();
    } else {
        info->GetOutput<TensorImpl>(0)->GetShape().Reshape(output_shape.GetDims(), output_shape.GetDimCount());
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
