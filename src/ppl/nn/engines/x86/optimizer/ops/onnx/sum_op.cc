#include "ppl/nn/engines/x86/optimizer/ops/onnx/sum_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/sum_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_sum.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SumOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeSum(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* SumOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<SumKernel>();
}

}}} // namespace ppl::nn::x86
