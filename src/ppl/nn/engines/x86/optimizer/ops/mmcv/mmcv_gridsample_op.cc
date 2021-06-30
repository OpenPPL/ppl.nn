#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_gridsample_op.h"
#include "ppl/nn/engines/x86/kernels/mmcv/mmcv_gridsample_kernel.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_gridsample.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MMCVGridSampleOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMMCVGridSample(info, param_.get());
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(DATATYPE_FLOAT32);
    };

    return RC_SUCCESS;
}

KernelImpl* MMCVGridSampleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MMCVGridSampleKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
