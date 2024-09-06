#ifdef PPLNN_ENABLE_FP8

#include "online_cast_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/opmx/f8f8/online_cast_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

RetCode F8F8OnlineCastOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(DATATYPE_FLOAT8E4M3);

        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->Reshape(input_shape->GetDims(), input_shape->GetDimCount());

        return RC_SUCCESS;
    };
    return RC_SUCCESS;
}

RetCode F8F8OnlineCastOp::DoInit(const OptKernelOptions& options) {
    return CommonInit();
}

KernelImpl* F8F8OnlineCastOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<F8F8OnlineCastKernel>();
}

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif