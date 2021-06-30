#include "ppl/nn/engines/x86/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/transpose_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_transpose.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode TransposeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeTranspose(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode TransposeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                  vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() ==
            DATAFORMAT_N16CX && // actually change N16CHW -> NHWC
        info.GetInput<TensorImpl>(0)->GetShape().GetDataType() == DATATYPE_FLOAT32 &&
        param_->perm == std::vector<int32_t>{0, 2, 3, 1} && param_->reverse == false) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    }
    return RC_SUCCESS;
}

KernelImpl* TransposeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<TransposeKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
