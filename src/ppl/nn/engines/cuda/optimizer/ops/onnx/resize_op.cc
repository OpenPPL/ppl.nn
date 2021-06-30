#include "ppl/nn/engines/cuda/optimizer/ops/onnx/resize_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/resize_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_resize.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ResizeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ResizeParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        auto status = type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
        if (info->GetInputCount() == 4) {
            auto shape = &info->GetInput<TensorImpl>(3)->GetShape();
            shape->SetDataType(ppl::common::DATATYPE_INT64);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        float* roi_data = nullptr;
        float* scales_data = nullptr;
        int64_t* sizes_data = nullptr;

        if (!info->GetInput<TensorImpl>(1)->GetShape().IsEmpty()) {
            auto shape = info->GetInput<TensorImpl>(1)->GetShape();
            roi_data = (float*)malloc(shape.GetBytesIncludingPadding());
            if (info->GetInput<TensorImpl>(1)->GetBufferPtr<void>() == nullptr)
                return RC_INVALID_VALUE;
            auto status = info->GetInput<TensorImpl>(1)->CopyToHost(roi_data);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy input 1 failed: " << GetRetCodeStr(status);
                return status;
            }
        }
        if (!info->GetInput<TensorImpl>(2)->GetShape().IsEmpty()) {
            auto shape = info->GetInput<TensorImpl>(2)->GetShape();
            scales_data = (float*)malloc(shape.GetBytesIncludingPadding());
            if (info->GetInput<TensorImpl>(2)->GetBufferPtr<void>() == nullptr) {
                return RC_INVALID_VALUE;
            }
            auto status = info->GetInput<TensorImpl>(2)->CopyToHost(scales_data);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy input 2 failed: " << GetRetCodeStr(status);
                return status;
            }
        }
        if (info->GetInputCount() == 4) {
            if (!info->GetInput<TensorImpl>(3)->GetShape().IsEmpty()) {
                auto shape = info->GetInput<TensorImpl>(3)->GetShape();
                sizes_data = (int64_t*)malloc(shape.GetBytesIncludingPadding());
                if (info->GetInput<TensorImpl>(3)->GetBufferPtr<void>() == nullptr) {
                    return RC_INVALID_VALUE;
                }
                auto status = info->GetInput<TensorImpl>(3)->CopyToHost(sizes_data);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Copy input 3 failed: " << GetRetCodeStr(status);
                    return status;
                }
            }
        }

        auto status = oputils::ReshapeResize(info, &param_, roi_data, scales_data, sizes_data);

        // release memory
        if (roi_data)
            free(roi_data);
        if (scales_data)
            free(scales_data);
        if (sizes_data)
            free(sizes_data);

        return status;
    };

    return RC_SUCCESS;
}

RetCode ResizeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

KernelImpl* ResizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ResizeKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
