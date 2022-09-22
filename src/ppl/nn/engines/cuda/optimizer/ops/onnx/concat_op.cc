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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/concat_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/concat_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_concat.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/concat.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace cuda {

ConcatOp::ConcatOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferHighestType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConcat(info, &param_.param);
    };

    infer_unsafe_dims_func_ = [](InputOutputInfo* info, std::set<uint32_t>* illegal_inputs) -> RetCode {
        auto in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetInputCount(); ++i) {
            if (illegal_inputs->find(i) != illegal_inputs->end()) {
                in_shape0 = info->GetInput<TensorImpl>(i)->GetShape();
                break;
            }
        }

        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->Reshape(in_shape0->GetDims(), in_shape0->GetRealDimCount());
        }
        return ppl::common::RC_SUCCESS;
    };
}

RetCode ConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ConcatParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

RetCode ConcatOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConcatKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
static RetCode SerializePrivateData(const pmx::SerializationContext& ctx, const ConcatExtraParam& extra_param, flatbuffers::FlatBufferBuilder* builder) {
    auto fb_concat_param = pmx::cuda::CreateConcatParam(*builder, extra_param.mask);
    auto fb_op_param = pmx::cuda::CreateOpParam(*builder, pmx::cuda::OpParamType_ConcatParam, fb_concat_param.Union());
    pmx::cuda::FinishOpParamBuffer(*builder, fb_op_param);
    return RC_SUCCESS;
}

static RetCode DeserializePrivateData(const void* fb_param, uint64_t size, ConcatExtraParam* extra_param) {
    auto fb_op_param = pmx::cuda::GetOpParam(fb_param);
    auto fb_concat_param = fb_op_param->value_as_ConcatParam();
    extra_param->mask = fb_concat_param->mask();
    return RC_SUCCESS;
}

RetCode ConcatOp::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder private_data_builder;
    auto status = SerializePrivateData(ctx, param_.extra_param, &private_data_builder);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SerializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::onnx::SerializeConcatParam(param_.param, &builder);
    auto fb_data = builder.CreateVector(private_data_builder.GetBufferPointer(), private_data_builder.GetSize());
    auto fb_root = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_ConcatParam, fb_param.Union(), fb_data);
    pmx::onnx::FinishOpParamBuffer(builder, fb_root);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

RetCode ConcatOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::onnx::GetOpParam(base);
    auto fb_concat_param = fb_op_param->value_as_ConcatParam();

    pmx::onnx::DeserializeConcatParam(*fb_concat_param, &param_.param);

    auto fb_data = fb_op_param->data_();
    auto status = DeserializePrivateData(fb_data->data(), fb_data->size(), &param_.extra_param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DeserializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    
    return RC_SUCCESS;
}
#endif
}}} // namespace ppl::nn::cuda
