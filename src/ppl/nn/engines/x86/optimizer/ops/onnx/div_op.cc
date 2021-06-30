#include "ppl/nn/engines/x86/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/div_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_add.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode DivOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeAdd(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode DivOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                            vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX ||
        info.GetInput<TensorImpl>(1)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_input_formats->at(1) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* DivOp::CreateKernelImpl() const {
    auto kernel = CreateKernelImplWithoutParam<DivKernel>();
    if (kernel) {
        kernel->SetFuseReLU(fuse_relu_);
    }
    return kernel;
}

}}} // namespace ppl::nn::x86
