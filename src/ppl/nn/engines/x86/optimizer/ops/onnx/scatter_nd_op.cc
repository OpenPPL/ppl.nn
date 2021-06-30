#include "ppl/nn/engines/x86/optimizer/ops/onnx/scatter_nd_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/scatter_nd_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_scatter_nd.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ScatterNDOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeScatterND(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* ScatterNDOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ScatterNdKernel>();
}

}}} // namespace ppl::nn::x86
