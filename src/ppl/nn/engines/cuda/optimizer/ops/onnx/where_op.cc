#include "ppl/nn/engines/cuda/optimizer/ops/onnx/where_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/where_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_where.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode WhereOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto shape = &info->GetInput<TensorImpl>(0)->GetShape();
        shape->SetDataType(ppl::common::DATATYPE_BOOL);

        auto in_shape = &info->GetInput<TensorImpl>(1)->GetShape();
        if (in_shape->GetDataType() == ppl::common::DATATYPE_UNKNOWN)
            return ppl::common::RC_UNSUPPORTED;
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataType(in_shape->GetDataType());
        }
        return ppl::common::RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeWhere(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode WhereOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* WhereOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<WhereKernel>();
}

}}} // namespace ppl::nn::cuda
