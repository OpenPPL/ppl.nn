#include "ppl/nn/engines/x86/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/resize_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_resize.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ResizeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeResize(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode ResizeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                               vector<dataformat_t>* selected_output_formats) {
    auto input_format = info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat();
    if (input_format == DATAFORMAT_N16CX &&
        param_->mode != ppl::nn::common::ResizeParam::RESIZE_MODE_CUBIC) { // cubic only support ndarray now
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }

    return RC_SUCCESS;
}

KernelImpl* ResizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ResizeKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
