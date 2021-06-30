#include "ppl/nn/engines/x86/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/relu_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ReluOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

RetCode ReluOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* ReluOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReluKernel>();
}

}}} // namespace ppl::nn::x86
