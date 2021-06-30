#include "ppl/nn/engines/x86/optimizer/ops/onnx/max_pool_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/maxpool_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pooling.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MaxPoolOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load outer_param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapePooling(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode MaxPoolOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX && info.GetOutputCount() == 1) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* MaxPoolOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MaxPoolKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
