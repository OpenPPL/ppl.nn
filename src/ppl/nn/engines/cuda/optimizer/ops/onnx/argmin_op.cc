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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/argmin_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/argmin_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_argmin.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

// #ifdef PPLNN_ENABLE_PMX_MODEL
// #include "ppl/nn/models/pmx/utils.h"
// #include "ppl/nn/models/pmx/oputils/onnx/argmin.h"
// #endif

namespace ppl { namespace nn { namespace cuda {

RetCode ArgminOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ArgMinParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

ArgminOp::ArgminOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto shape = info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(ppl::common::DATATYPE_INT64);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeArgMin(info, &param_);
    };
}

RetCode ArgminOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ArgminOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ArgMinKernel>(&param_);
}

// #ifdef PPLNN_ENABLE_PMX_MODEL
//     ppl::common::RetCode ArgminOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
//         flatbuffers::FlatBufferBuilder builder;
//         auto fb_param = pmx::onnx::SerializeArgMinParam(param_, &builder);
//         auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_ArgMinParam, fb_param.Union());
//         pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
//         return ds->Write(builder.GetBufferPointer(), builder.GetSize());
//     }
//     ppl::common::RetCode ArgminOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
//         auto fb_op_param = pmx::onnx::GetOpParam(base);
//         auto fb_argmin_param = fb_op_param->value_as_ArgMinParam();
//         pmx::onnx::DeserializeArgMinParam(*fb_argmin_param, &param_);
//         return ppl::common::RC_SUCCESS;
//     }
// #endif
}}} // namespace ppl::nn::cuda
