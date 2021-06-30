#include "ppl/nn/engines/x86/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/concat_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_concat.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeConcat(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode ConcatOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                               vector<dataformat_t>* selected_output_formats) {
    const uint32_t input_count = info.GetInputCount();
    bool input_all_16c = true;
    for (uint32_t i = 0; i < input_count; i++) {
        if (info.GetInput<TensorImpl>(i)->GetShape().GetDataFormat() != DATAFORMAT_N16CX) {
            input_all_16c = false;
        }
    }

    if (input_all_16c) {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_formats->at(i) = DATAFORMAT_N16CX;
        }
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }

    return RC_SUCCESS;
}

KernelImpl* ConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConcatKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
