#include "ppl/nn/engines/x86/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/engines/x86/kernels/ppl/reorder_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ReorderOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input = info->GetInput<TensorImpl>(0)->GetShape();
        auto& output = info->GetOutput<TensorImpl>(0)->GetShape();
        if (output.GetDataFormat() == DATAFORMAT_N16CX && input.GetDimCount() < 3) {
            auto padded_output_shape = PadShapeTo3Dims(input);
            output.Reshape(padded_output_shape.GetDims(), padded_output_shape.GetDimCount());
        } else {
            if (input.IsScalar()) {
                output.ReshapeAsScalar();
            } else {
                output.Reshape(input.GetDims(), input.GetDimCount());
            }
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* ReorderOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReorderKernel>();
}

}}} // namespace ppl::nn::x86
