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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/pad_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/pad_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pad.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/pad.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode PadOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<PadParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

PadOp::PadOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }

        if (info->GetInputCount() >= 2) {
            auto shape = info->GetInput<TensorImpl>(1)->GetShape();
            shape->SetDataType(DATATYPE_INT64);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        const TensorShape& shape = *info->GetInput<TensorImpl>(0)->GetShape();
        uint32_t dim_count = shape.GetDimCount();

        if (info->GetInputCount() == 1) {
            return onnx::ReshapePad(info, &param_);
        } else {
            auto pad = info->GetInput<TensorImpl>(1);
            if (pad->GetShape()->GetDimCount() != 1 || pad->GetShape()->GetDim(0) != 2 * dim_count ||
                pad->GetShape()->GetDataType() != DATATYPE_INT64) {
                return RC_INVALID_VALUE;
            }
            int pad_elems = pad->GetShape()->CalcElementsIncludingPadding();
            vector<int64_t> pad_data(pad_elems);
            for (int it = 0; it < pad_elems; pad_data[it] = 0, ++it)
                ;
            if (pad->GetBufferPtr() != nullptr) {
                auto status = pad->CopyToHost(pad_data.data());
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Copy pad data failed: " << GetRetCodeStr(status);
                    return status;
                }
            }

            return onnx::ReshapePad(info, &param_, pad_data.data(), pad_data.data() + dim_count);
        }
    };

}

RetCode PadOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* PadOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PadKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode PadOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializePadParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_PadParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode PadOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_PadParam();
        pmx::onnx::DeserializePadParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
