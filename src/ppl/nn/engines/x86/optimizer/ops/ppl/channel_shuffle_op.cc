#include "ppl/nn/engines/x86/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/x86/kernels/ppl/channel_shuffle_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ChannelShuffleOp::Init(const OptKernelOptions& options) {
    if (options.graph_data) {
        auto status = GenericLoadParam(options, &param_);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
            return status;
        }
    } else {
        param_ = make_shared<ppl::nn::common::ChannelShuffleParam>();
    }
    infer_type_func_ = GenericInferType;
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

void ChannelShuffleOp::SetGroup(int group) {
    param_->group = group;
}

RetCode ChannelShuffleOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    auto input_format = info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat();

    if (input_format == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }

    if (input_format == DATAFORMAT_NDARRAY) {
        selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
        selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    }

    return RC_SUCCESS;
}

KernelImpl* ChannelShuffleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ChannelShuffleKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
