#include "ppl/nn/engines/cuda/optimizer/ops/onnx/exp_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/exp_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ExpOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        type = ppl::common::DATATYPE_FLOAT32;
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode ExpOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ExpOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ExpKernel>();
}

}}} // namespace ppl::nn::cuda
