#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeGemm(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() < 2) {
        LOG(ERROR) << "2 input required at least.";
        return RC_INVALID_VALUE;
    }

    auto param = (const GemmParam*)arg;
    auto A = &info->GetInput<TensorImpl>(0)->GetShape();
    auto B = &info->GetInput<TensorImpl>(1)->GetShape();
    auto Y = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (A->GetDimCount() != 2 || B->GetDimCount() != 2) {
        LOG(ERROR) << "A and B must be 2D tensor.";
    }

    int32_t AMdim = 0;
    int32_t BNdim = 1;
    if (param->transA) {
        AMdim = 1;
    }
    if (param->transB) {
        BNdim = 0;
    }

    Y->Reshape({A->GetDim(AMdim), param->N == 0 ? B->GetDim(BNdim) : param->N});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
