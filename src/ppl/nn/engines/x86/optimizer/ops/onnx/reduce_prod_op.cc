#include "ppl/nn/engines/x86/optimizer/ops/onnx/reduce_prod_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/reduce_prod_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reduce.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ReduceProdOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeReduce(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode ReduceProdOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                   vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX && param_->keep_dims) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* ReduceProdOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ReduceProdKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
