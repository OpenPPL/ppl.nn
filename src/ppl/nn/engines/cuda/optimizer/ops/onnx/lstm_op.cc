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
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode LstmOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<LSTMParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        //if (type == DATATYPE_INT8) {
        //    auto in_edge_id = info->GetInput<TensorImpl>(0)->GetEdge()->GetId();
        //    auto& in_quant = quant->at(in_edge_id);
        //    auto out_edge_id = info->GetOutput<TensorImpl>(0)->GetEdge()->GetId();
        //    auto& out_quant = quant->at(out_edge_id);
        //    if (in_quant.type != DATATYPE_INT8 || out_quant.type != DATATYPE_INT8) {
        //        return RC_INVALID_VALUE;
        //    }
        //    info->GetInput<TensorImpl>(0)->GetShape().SetDataType(in_quant.type);
        //    info->GetOutput<TensorImpl>(0)->GetShape().SetDataType(out_quant.type);

        //    // Copy quant info skipping input0
        //    for (uint32_t i = 1; i < info->GetInputCount(); ++i) {
        //        auto in_edge_id = info->GetInput<TensorImpl>(i)->GetEdge()->GetId();
        //        auto& in_quant = quant->at(in_edge_id);
        //        auto in_shape = &info->GetInput<TensorImpl>(i)->GetShape();
        //        if (i == 1 && in_quant.type != DATATYPE_UNKNOWN) {
        //            in_shape->SetDataType(in_quant.type);
        //            continue;
        //        }
        //        if (i == 2 && param_.param.bias_term) {
        //            in_shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
        //            continue;
        //        }
        //        in_quant = out_quant;
        //        in_shape->SetDataType(out_quant.type);
        //    }
        //    return ppl::common::RC_SUCCESS;
        //}
        type = ppl::common::DATATYPE_FLOAT16;
        return InferDefaultType(info, type);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeLSTM(info, &param_);
    };

    return RC_SUCCESS;
}

RetCode LstmOp::Finalize(const OptKernelOptions& options) {
    param_ = *((LSTMParam*)options.param);

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

}}} // namespace ppl::nn::cuda
