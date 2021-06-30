#include "ppl/nn/engines/x86/optimizer/ops/onnx/clip_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/clip_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ClipOp::Init(const OptKernelOptions&) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

RetCode ClipOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* ClipOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ClipKernel>();
}

}}} // namespace ppl::nn::x86
