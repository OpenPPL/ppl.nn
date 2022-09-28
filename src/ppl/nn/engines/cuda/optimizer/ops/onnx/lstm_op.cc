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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/lstm_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/lstm_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_lstm.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {

RetCode LstmOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<LSTMParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

LstmOp::LstmOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        type = ppl::common::DATATYPE_FLOAT16;
        return InferDefaultType(info, type);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeLSTM(info, &param_);
    };
}

RetCode LstmOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

void LstmOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new LSTMParam();
    }
    *(LSTMParam*)param = param_;
    return;
}

KernelImpl* LstmOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LstmKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LstmOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializeLSTMParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_LSTMParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode LstmOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_LSTMParam();
        pmx::onnx::DeserializeLSTMParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
