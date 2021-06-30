#include "ppl/nn/engines/x86/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/sub_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_add.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SubOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeAdd(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode SubOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                            vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX ||
        info.GetInput<TensorImpl>(1)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_input_formats->at(1) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* SubOp::CreateKernelImpl() const {
    auto kernel = CreateKernelImplWithoutParam<SubKernel>();
    if (kernel) {
        kernel->SetFuseReLU(fuse_relu_);
    }
    return kernel;
}

}}} // namespace ppl::nn::x86
