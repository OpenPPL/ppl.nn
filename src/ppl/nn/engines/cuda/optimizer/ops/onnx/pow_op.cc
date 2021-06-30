#include "ppl/nn/engines/cuda/optimizer/ops/onnx/pow_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/pow_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_add.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode PowOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeAdd(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode PowOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* PowOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<PowKernel>();
}

}}} // namespace ppl::nn::cuda
