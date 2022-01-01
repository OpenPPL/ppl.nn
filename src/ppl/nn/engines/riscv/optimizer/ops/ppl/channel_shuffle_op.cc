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

#include "ppl/nn/engines/riscv/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/riscv/kernels/ppl/channel_shuffle_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ChannelShuffleOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
    }

    infer_type_func_ = GenericInferType;

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input0 = *info->GetInput<TensorImpl>(0)->GetShape();
        int64_t channels = input0.GetDim(1);
        for (uint32_t i = 1; i < info->GetInputCount(); ++i) {
            channels += info->GetInput<TensorImpl>(1)->GetShape()->GetDim(1);
        }
        if (channels % info->GetOutputCount()) {
            return ppl::common::RC_INVALID_VALUE;
        }
        channels /= info->GetOutputCount();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto& output = *info->GetOutput<TensorImpl>(i)->GetShape();
            output.Reshape(input0.GetDims(), input0.GetRealDimCount());
            output.SetDim(1, channels);
        }

        return RC_SUCCESS;
    };

    return RC_SUCCESS;
}

void ChannelShuffleOp::SetGroup(int64_t group) {
    param_->group = group;
}

RetCode ChannelShuffleOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    if (info.GetInputCount() == 2) {
        auto input_format1 = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
        auto input_format2 = info.GetInput<TensorImpl>(1)->GetShape()->GetDataFormat();
        if (DATAFORMAT_N4CX == input_format1 && DATAFORMAT_N4CX == input_format2) {
            selected_input_formats->at(0) = DATAFORMAT_N4CX;
            selected_input_formats->at(1) = DATAFORMAT_N4CX;
            selected_output_formats->at(0) = DATAFORMAT_N4CX;
            if (info.GetOutputCount() == 2) {
                selected_output_formats->at(1) = DATAFORMAT_N4CX;
            }
        } else if (DATAFORMAT_N8CX == input_format1 && DATAFORMAT_N8CX == input_format2) {
            selected_input_formats->at(0) = DATAFORMAT_N8CX;
            selected_input_formats->at(1) = DATAFORMAT_N8CX;
            selected_output_formats->at(0) = DATAFORMAT_N8CX;
            if (info.GetOutputCount() == 2) {
                selected_output_formats->at(1) = DATAFORMAT_N8CX;
            }
        } else {
            selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
            selected_input_formats->at(1) = DATAFORMAT_NDARRAY;
            selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
            if (info.GetOutputCount() == 2) {
                selected_output_formats->at(1) = DATAFORMAT_NDARRAY;
            }
        }
    } else {
        auto input_format = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
        if (DATAFORMAT_N4CX == input_format) {
            selected_input_formats->at(0) = DATAFORMAT_N4CX;
            selected_output_formats->at(0) = DATAFORMAT_N4CX;
        } else if (DATAFORMAT_N8CX == input_format) {
            selected_input_formats->at(0) = DATAFORMAT_N8CX;
            selected_output_formats->at(0) = DATAFORMAT_N8CX;
        } else {
            selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
            selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
        }
    }

    return RC_SUCCESS;
}

RetCode ChannelShuffleOp::SelectDataType(const InputOutputInfo& info, vector<datatype_t>* selected_input_data_types,
                                         vector<datatype_t>* selected_output_data_types) {
    if (info.GetInputCount() == 2) {
        auto input_datatype1 = info.GetInput<TensorImpl>(0)->GetShape()->GetDataType();
        auto input_datatype2 = info.GetInput<TensorImpl>(1)->GetShape()->GetDataType();
        if (DATATYPE_FLOAT32 == input_datatype1 && DATATYPE_FLOAT32 == input_datatype2) {
            selected_input_data_types->at(0) = DATATYPE_FLOAT32;
            selected_input_data_types->at(1) = DATATYPE_FLOAT32;
            selected_output_data_types->at(0) = DATATYPE_FLOAT32;
            if (info.GetOutputCount() == 2) {
                selected_output_data_types->at(1) = DATATYPE_FLOAT32;
            }
        } else if (DATATYPE_FLOAT16 == input_datatype1 && DATATYPE_FLOAT16 == input_datatype2) {
            selected_input_data_types->at(0) = DATATYPE_FLOAT16;
            selected_input_data_types->at(1) = DATATYPE_FLOAT16;
            selected_output_data_types->at(0) = DATATYPE_FLOAT16;
            if (info.GetOutputCount() == 2) {
                selected_output_data_types->at(1) = DATATYPE_FLOAT16;
            }
        } else {
            LOG(ERROR) << "unsupported datatypes: " << ppl::common::GetDataTypeStr(input_datatype1) << " && "
                       << ppl::common::GetDataTypeStr(input_datatype2);
        }
    } else {
        auto input_datatype = info.GetInput<TensorImpl>(0)->GetShape()->GetDataType();
        if (DATATYPE_FLOAT32 == input_datatype) {
            selected_input_data_types->at(0) = DATATYPE_FLOAT32;
            selected_output_data_types->at(0) = DATATYPE_FLOAT32;
        } else if (DATATYPE_FLOAT16 == input_datatype) {
            selected_input_data_types->at(0) = DATATYPE_FLOAT16;
            selected_output_data_types->at(0) = DATATYPE_FLOAT16;
        } else {
            LOG(ERROR) << "unsupported datatypes: " << ppl::common::GetDataTypeStr(input_datatype);
        }
    }

    return RC_SUCCESS;
}

KernelImpl* ChannelShuffleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ChannelShuffleKernel>(param_.get());
}

}}} // namespace ppl::nn::riscv
