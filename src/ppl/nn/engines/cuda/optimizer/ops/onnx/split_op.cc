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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/split_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/split_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_split.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/split.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode SplitOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<SplitParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph->data;

    auto split_data_it = graph_data->constants.find(node->GetInput(1));
    const int64_t* split_data = nullptr;
    if (split_data_it != graph_data->constants.end()) {
        split_data = (const int64_t*)split_data_it->second.data.GetData();
    }

    if (split_data != nullptr) {
        auto shape_shape_it = graph_data->shapes.find(node->GetInput(1));
        if (shape_shape_it != graph_data->shapes.end()) {
            auto& shape_shape = shape_shape_it->second;
            constant_split_data_.assign(split_data, split_data + shape_shape.dims[0]);
        }
    }

    return RC_SUCCESS;
}

SplitOp::SplitOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() > 2) {
            LOG(ERROR) << "invalid input size.";
            return RC_INVALID_VALUE;
        }

        if (info->GetInputCount() == 1) {
            auto in_shape = info->GetInput<TensorImpl>(0)->GetShape();
            info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(in_shape->GetDims(), in_shape->GetRealDimCount());
            return RC_SUCCESS;
        } else {
            if (constant_split_data_.empty()) {
                auto input = info->GetInput<TensorImpl>(1);
                if (!input->GetBufferPtr()) {
                    return RC_NOT_FOUND;
                }
                const TensorShape& dst_desc = *input->GetShape();
                if (dst_desc.CalcElementsIncludingPadding() == 0) {
                    auto in_shape = info->GetInput<TensorImpl>(0)->GetShape();
                    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(in_shape->GetDims(), in_shape->GetRealDimCount());
                    return RC_SUCCESS;
                }

                vector<int64_t> split_data(dst_desc.CalcElementsIncludingPadding());
                auto status = input->CopyToHost(split_data.data());
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Copy shape data failed: " << GetRetCodeStr(status);
                    return status;
                }
                return onnx::ReshapeSplit(info, &param_, split_data.data());
            } else {
                return onnx::ReshapeSplit(info, &param_, constant_split_data_.data());
            }
        }
    };
}

RetCode SplitOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* SplitOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SplitKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode SplitOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializeSplitParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_SplitParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode SplitOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_SplitParam();
        pmx::onnx::DeserializeSplitParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
