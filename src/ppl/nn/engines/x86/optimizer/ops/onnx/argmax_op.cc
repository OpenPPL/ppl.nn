#include "ppl/nn/engines/x86/optimizer/ops/onnx/argmax_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/argmax_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_argmax.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ArgmaxOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeArgMax(info, param_.get());
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

KernelImpl* ArgmaxOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ArgMaxKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
