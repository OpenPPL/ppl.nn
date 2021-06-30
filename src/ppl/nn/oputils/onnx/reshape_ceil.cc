#include "ppl/nn/oputils/onnx/reshape_ceil.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeCeil(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input = &info->GetInput<TensorImpl>(0)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    output->SetDataType(DATATYPE_FLOAT32);
    if (input->IsScalar()) {
        output->ReshapeAsScalar();
    } else {
        output->Reshape(input->GetDims(), input->GetDimCount());
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
