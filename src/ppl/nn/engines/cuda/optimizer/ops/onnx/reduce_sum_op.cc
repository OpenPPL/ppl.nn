#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reduce_sum_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/reduce_sum_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reduce.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ReduceSumOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ReduceParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeReduce(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode ReduceSumOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ReduceSumOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ReduceSumKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
