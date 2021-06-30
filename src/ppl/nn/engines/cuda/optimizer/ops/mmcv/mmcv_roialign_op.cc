#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_roialign_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_roialign_kernel.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_roialign.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode MMCVROIAlignOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<MMCVROIAlignParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        for (uint32_t i = 0; i < info->GetInputCount(); ++i) {
            auto in_shape = &info->GetInput<TensorImpl>(i)->GetShape();
            if (in_shape->GetDataType() == DATATYPE_UNKNOWN) {
                return RC_UNSUPPORTED;
            }
            if (in_shape->GetDataType() == DATATYPE_FLOAT16) {
                in_shape->SetDataType(DATATYPE_FLOAT32);
            }
        }
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataType(DATATYPE_FLOAT32);
        }
        return ppl::common::RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMMCVROIAlign(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode MMCVROIAlignOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* MMCVROIAlignOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MMCVROIAlignKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
