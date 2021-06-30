#include "ppl/nn/engines/common/ppl/converter_kernel.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/utils/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

RetCode ConverterKernel::DoExecute(KernelExecContext* ctx) {
    if (ctx->GetInputCount() != ctx->GetOutputCount()) {
        LOG(ERROR) << "input count [" << ctx->GetInputCount() << "] != output count [" << ctx->GetOutputCount() << "]";
        return RC_INVALID_VALUE;
    }

    utils::GenericCpuDevice tmp_cpu_device;
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto src = ctx->GetInput<TensorImpl>(i);
        auto dst = ctx->GetOutput<TensorImpl>(i);

        auto src_barrier = ctx->GetInputBarrier(i);
        if (src_barrier) {
            auto status = src_barrier->Sync();
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "sync EdgeObject[" << src->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }

        dst->GetShape() = src->GetShape();
        dst->GetShape().SetDataFormat(DATAFORMAT_NDARRAY);

        auto status = utils::CopyTensorBuffer(*src, dst, &tmp_cpu_device);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy tensor from [" << src->GetName() << "] to [" << dst->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::common
