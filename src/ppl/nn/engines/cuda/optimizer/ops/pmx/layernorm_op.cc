#include "ppl/nn/engines/cuda/optimizer/ops/pmx/layernorm_op.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/layernorm_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace cuda {

RetCode LayerNormOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<LayerNormParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

LayerNormOp::LayerNormOp(const ir::Node* node) : CudaOptKernel(node) {

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        // type = ppl::common::DATATYPE_FLOAT32; // only support fp32 for now
        type = info->GetInput<TensorImpl>(0)->GetShape()->GetDataType();
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
            for (uint32_t i = 1; i < 3; ++i) {
                if (info->GetInputCount() > i) {
                    auto shape = info->GetInput<TensorImpl>(i)->GetShape();
                    shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
                }
            }
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (param_.elementwise_affine && info->GetInputCount() != 3) {
            LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount()
                        << "] != 3 when elementwise_affine is True";
            return RC_INVALID_VALUE;
        }
        return GenericInferDims(info);
    };

}
RetCode LayerNormOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* LayerNormOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LayerNormKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
