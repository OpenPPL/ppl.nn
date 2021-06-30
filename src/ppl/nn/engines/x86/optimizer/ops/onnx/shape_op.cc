#include "ppl/nn/engines/x86/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/shape_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ShapeOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto output_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->Reshape({info->GetInput<TensorImpl>(0)->GetShape().GetRealDimCount()});
        return RC_SUCCESS;
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        auto output_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

RetCode ShapeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                              vector<dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat();
    return RC_SUCCESS;
}

KernelImpl* ShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ShapeKernel>();
}

}}} // namespace ppl::nn::x86
