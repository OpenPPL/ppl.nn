#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_non_max_suppression_op.h"
#include "ppl/nn/engines/x86/kernels/mmcv/mmcv_non_max_suppression_kernel.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_non_max_suppression.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MMCVNonMaxSuppressionOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMMCVNonMaxSuppression(info, param_.get());
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

KernelImpl* MMCVNonMaxSuppressionOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MMCVNonMaxSuppressionKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
