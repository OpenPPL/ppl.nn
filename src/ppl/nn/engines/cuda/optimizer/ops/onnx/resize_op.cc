// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/resize_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/resize_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_resize.h"
#include <iostream>

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/resize.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode ResizeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ResizeParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

ResizeOp::ResizeOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        auto shape = info->GetInput<TensorImpl>(2)->GetShape();
        shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
        if (info->GetInputCount() == 4) {
            auto shape = info->GetInput<TensorImpl>(3)->GetShape();
            shape->SetDataType(ppl::common::DATATYPE_INT64);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        float* roi_data = nullptr;
        float* scales_data = nullptr;
        int64_t* sizes_data = nullptr;

        if (info->GetInputCount() >= 2 && info->GetInput<TensorImpl>(1) != nullptr) {
            if (!info->GetInput<TensorImpl>(1)->GetShape()->IsEmpty()) {
                const TensorShape& shape = *info->GetInput<TensorImpl>(1)->GetShape();
                roi_data = (float*)malloc(shape.CalcBytesIncludingPadding());
                if (info->GetInput<TensorImpl>(1)->GetBufferPtr<void>() == nullptr)
                    return RC_INVALID_VALUE;
                auto status = info->GetInput<TensorImpl>(1)->CopyToHost(roi_data);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Copy input 1 failed: " << GetRetCodeStr(status);
                    return status;
                }
            }
        }
        if (info->GetInputCount() >= 3 && info->GetInput<TensorImpl>(2) != nullptr) {
            if (!info->GetInput<TensorImpl>(2)->GetShape()->IsEmpty()) {
                const TensorShape& shape = *info->GetInput<TensorImpl>(2)->GetShape();
                scales_data = (float*)malloc(shape.CalcBytesIncludingPadding());
                if (info->GetInput<TensorImpl>(2)->GetBufferPtr<void>() == nullptr) {
                    return RC_INVALID_VALUE;
                }
                auto status = info->GetInput<TensorImpl>(2)->CopyToHost(scales_data);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Copy input 2 failed: " << GetRetCodeStr(status);
                    return status;
                }
            }
        }
        if (info->GetInputCount() >= 4 && info->GetInput<TensorImpl>(3) != nullptr) {
            if (!info->GetInput<TensorImpl>(3)->GetShape()->IsEmpty()) {
                const TensorShape& shape = *info->GetInput<TensorImpl>(3)->GetShape();
                sizes_data = (int64_t*)malloc(shape.CalcBytesIncludingPadding());
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

        auto status = onnx::ReshapeResize(info, &param_, roi_data, scales_data, sizes_data);

        // release memory
        if (roi_data)
            free(roi_data);
        if (scales_data)
            free(scales_data);
        if (sizes_data)
            free(sizes_data);

        return status;
    };
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

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode ResizeOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializeResizeParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_ResizeParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode ResizeOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_ResizeParam();
        pmx::onnx::DeserializeResizeParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
