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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/squeeze_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/squeeze_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_squeeze.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/squeeze.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode SqueezeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<SqueezeParam>(options, &param_);
    if (status != RC_SUCCESS && GetNode()->GetType().version < 13) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

SqueezeOp::SqueezeOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        if (info->GetInputCount() > 1) {
            auto shape = info->GetInput<TensorImpl>(1)->GetShape();
            shape->SetDataType(DATATYPE_INT64);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() > 1) {
            auto axes_input = info->GetInput<TensorImpl>(1);
            auto size = axes_input->GetShape()->CalcElementsExcludingPadding();
            this->param_.axes.resize(size);
            auto axes_data = (int64_t*)malloc(size * sizeof(int64_t));
            axes_input->CopyToHost(axes_data);
            for (uint32_t i = 0; i < size; ++i) {
                this->param_.axes[i] = axes_data[i];
            }
        }
        return onnx::ReshapeSqueeze(info, &param_);
    };
}

RetCode SqueezeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* SqueezeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SqueezeKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode SqueezeOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializeSqueezeParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_SqueezeParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode SqueezeOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_SqueezeParam();
        pmx::onnx::DeserializeSqueezeParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
