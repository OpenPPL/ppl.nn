#include "ppl/nn/engines/x86/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/range_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_range.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode RangeOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeRange(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* RangeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<RangeKernel>();
}

}}} // namespace ppl::nn::x86
