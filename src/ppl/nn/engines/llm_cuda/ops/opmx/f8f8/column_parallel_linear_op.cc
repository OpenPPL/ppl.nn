#ifdef PPLNN_ENABLE_FP8

#include "column_parallel_linear_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/opmx/f8f8/column_parallel_linear_kernel.h"
#include "ppl/nn/oputils/opmx/reshape_column_parallel_linear.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

RetCode F8F8ColumnParallelLinearOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(DATATYPE_FLOAT16);
        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return ppl::nn::opmx::ReshapeColumnParallelLinear(info, param_.get(), nccl_param_->size);
    };
    return RC_SUCCESS;
}


RetCode F8F8ColumnParallelLinearOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::opmx::ColumnParallelLinearParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    nccl_param_ = options.device->GetTensorParallelNcclParam();

    return CommonInit();
}

KernelImpl* F8F8ColumnParallelLinearOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<F8F8ColumnParallelLinearKernel>(param_.get());
}

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif