#include "ppl/nn/engines/cuda/optimizer/ops/onnx/transpose_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/transpose_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_transpose.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode TransposeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<TransposeParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();

        TransposeParam modified_param = param_;
        if (modified_param.perm.empty()) {
            int32_t dim_count = in_shape0.GetDimCount();
            modified_param.perm.resize(dim_count);
            for (int it = 0; it < dim_count; ++it) {
                modified_param.perm[it] = dim_count - it - 1;
            }
        }

        if (modified_param.perm.size() != in_shape0.GetDimCount()) {
            return RC_UNSUPPORTED;
        }

        return oputils::ReshapeTranspose(info, &modified_param);
    };

    return RC_SUCCESS;
}

RetCode TransposeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* TransposeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<TransposeKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
