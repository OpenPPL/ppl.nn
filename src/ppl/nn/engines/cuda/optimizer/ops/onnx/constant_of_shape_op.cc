#include "ppl/nn/engines/cuda/optimizer/ops/onnx/constant_of_shape_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_constant_of_shape.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ConstantOfShapeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ConstantOfShapeParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto output = &info->GetOutput<TensorImpl>(0)->GetShape();
        output->SetDataType(param_.data_type);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
            return RC_INVALID_VALUE;
        }

        auto input = info->GetInput<TensorImpl>(0);
        if (!input->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }

        std::unique_ptr<int64_t[]> input_host(new int64_t[input->GetShape().GetElementsIncludingPadding()]);
        auto status = input->CopyToHost(input_host.get());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Copy input host failed: " << GetRetCodeStr(status);
            return status;
        }

        return oputils::ReshapeConstantOfShape(info, &param_, input_host.get());
    };

    return RC_SUCCESS;
}

RetCode ConstantOfShapeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ConstantOfShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConstantOfShapeKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
