#include "ppl/nn/engines/x86/optimizer/ops/onnx/matmul_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/matmul_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_matmul.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MatMulOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMatMul(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* MatMulOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<MatMulKernel>();
}

}}} // namespace ppl::nn::x86
