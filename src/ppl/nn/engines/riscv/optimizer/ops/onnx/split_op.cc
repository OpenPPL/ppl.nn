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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/split_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_split.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode SplitOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto ret = oputils::ReshapeSplit(info, param_.get());
        if (ret != RC_SUCCESS) {
            return ret;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode SplitOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                              vector<dataformat_t>* selected_output_formats) {
    if (DATAFORMAT_N8CX == selected_input_formats->at(0)) {
        for (int32_t i = 0; i < selected_output_formats->size(); i++) {
            selected_output_formats->at(i) = DATAFORMAT_N8CX;
        }
    } else if (DATAFORMAT_N4CX == selected_input_formats->at(0)) {
        for (int32_t i = 0; i < selected_output_formats->size(); i++) {
            selected_output_formats->at(i) = DATAFORMAT_N4CX;
        }
    } else if (DATAFORMAT_NDARRAY == selected_input_formats->at(0)) {
        for (int32_t i = 0; i < selected_output_formats->size(); i++) {
            selected_output_formats->at(i) = DATAFORMAT_NDARRAY;
        }
    }

    return RC_SUCCESS;
}

RetCode SplitOp::SelectDataType(const InputOutputInfo& info, std::vector<datatype_t>* selected_input_data_types,
                                std::vector<datatype_t>* selected_output_data_types) {
    if (DATATYPE_FLOAT16 == selected_input_data_types->at(0)) {
        for (int32_t i = 0; i < selected_output_data_types->size(); i++) {
            selected_output_data_types->at(i) = DATATYPE_FLOAT16;
        }
    } else if (DATATYPE_FLOAT32 == selected_input_data_types->at(0)) {
        for (int32_t i = 0; i < selected_output_data_types->size(); i++) {
            selected_output_data_types->at(i) = DATATYPE_FLOAT32;
        }
    }

    return RC_SUCCESS;
}

KernelImpl* SplitOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SplitKernel>(param_.get());
}

}}} // namespace ppl::nn::riscv