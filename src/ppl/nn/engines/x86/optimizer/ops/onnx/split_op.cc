#include "ppl/nn/engines/x86/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/split_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_split.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SplitOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto ret = oputils::ReshapeSplit(info, param_.get());
        if (ret != RC_SUCCESS) {
            return ret;
        }
        if (info->GetInput<TensorImpl>(0)->GetShape().GetDataType() != DATATYPE_FLOAT32) {
            LOG(ERROR) << "only support fp32 now.";
            return RC_UNSUPPORTED;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode SplitOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                              vector<dataformat_t>* selected_output_formats) {
    auto input_format = info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat();
    if (input_format == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        for (uint32_t i = 0; i < info.GetOutputCount(); i++) {
            selected_output_formats->at(i) = DATAFORMAT_N16CX;
        }
    }

    return RC_SUCCESS;
}

KernelImpl* SplitOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SplitKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
