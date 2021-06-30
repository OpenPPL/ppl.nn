#include "ppl/nn/engines/x86/optimizer/ops/onnx/pow_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/pow_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_add.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode PowOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto ret = oputils::ReshapeAdd(info, nullptr);
        if (ret != RC_SUCCESS) {
            return ret;
        }

        if (info->GetOutput<TensorImpl>(0)->GetShape().GetDataType() != DATATYPE_FLOAT32) {
            LOG(ERROR) << "only support fp32 now.";
            return RC_UNSUPPORTED;
        }
        if (info->GetOutput<TensorImpl>(0)->GetShape().GetDataFormat() != DATAFORMAT_NDARRAY) {
            LOG(ERROR) << "only support ndarray now.";
            return RC_UNSUPPORTED;
        }
        if (info->GetOutput<TensorImpl>(0)->GetShape().GetDimCount() > 6) {
            LOG(ERROR) << "tensor's dim count must <= 6.";
            return RC_UNSUPPORTED;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* PowOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<PowKernel>();
}

}}} // namespace ppl::nn::x86
