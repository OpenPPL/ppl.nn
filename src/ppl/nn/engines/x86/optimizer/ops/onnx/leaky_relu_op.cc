#include "ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/leaky_relu_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_leaky_relu.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode LeakyReluOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeLeakyReLU(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode LeakyReluOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                  vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* LeakyReluOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LeakyReluKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
