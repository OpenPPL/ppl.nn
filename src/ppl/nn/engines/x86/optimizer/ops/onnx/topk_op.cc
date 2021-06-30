#include "ppl/nn/engines/x86/optimizer/ops/onnx/topk_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/topk_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_topk.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode TopKOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeTopK(info, param_.get());
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        GenericInferType(info);
        auto out_shape = &info->GetOutput<TensorImpl>(1)->GetShape();
        out_shape->SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

KernelImpl* TopKOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<TopKKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
