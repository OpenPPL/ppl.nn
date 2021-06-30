#include "ppl/nn/engines/x86/optimizer/ops/onnx/tile_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/tile_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_tile.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode TileOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto ret = oputils::ReshapeTile(info, nullptr);
        if (ret != RC_SUCCESS) {
            return ret;
        }
        auto A = &info->GetInput<TensorImpl>(0)->GetShape();
        auto B = &info->GetInput<TensorImpl>(1)->GetShape();
        auto C = &info->GetOutput<TensorImpl>(0)->GetShape();

        if (A->GetDataType() != DATATYPE_FLOAT32 && A->GetDataType() != DATATYPE_INT64) {
            LOG(ERROR) << "unsupported data type " << GetDataTypeStr(A->GetDataType());
            return RC_UNSUPPORTED;
        }
        if (A->GetDataFormat() != DATAFORMAT_NDARRAY || B->GetDataFormat() != DATAFORMAT_NDARRAY ||
            C->GetDataFormat() != DATAFORMAT_NDARRAY) {
            LOG(ERROR) << "only support ndarray now.";
            return RC_UNSUPPORTED;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* TileOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<TileKernel>();
}

}}} // namespace ppl::nn::x86
