#include "ppl/nn/engines/cuda/optimizer/ops/pmx/gelu_op.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/gelu_kernel.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {
RetCode GeluOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

GeluOp::GeluOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = GenericInferDims;
}

RetCode GeluOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* GeluOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<GeluKernel>();
}

}}} // namespace ppl::nn::cuda
