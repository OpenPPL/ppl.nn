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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/concat_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_concat.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConcat(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode ConcatOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                               vector<dataformat_t>* selected_output_formats) {
    const uint32_t input_count = info.GetInputCount();
    if (DATAFORMAT_N8CX == selected_input_formats->at(0)) {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_formats->at(i) = DATAFORMAT_N8CX;
        }
        selected_output_formats->at(0) = DATAFORMAT_N8CX;
    } else if (DATAFORMAT_N4CX == selected_input_formats->at(0)) {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_formats->at(i) = DATAFORMAT_N4CX;
        }
        selected_output_formats->at(0) = DATAFORMAT_N4CX;
    } else if (DATAFORMAT_N2CX == selected_input_formats->at(0)) {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_formats->at(i) = DATAFORMAT_N2CX;
        }
        selected_output_formats->at(0) = DATAFORMAT_N2CX;
    } else {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_formats->at(i) = DATAFORMAT_NDARRAY;
        }
        selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    }

    return RC_SUCCESS;
}

RetCode ConcatOp::SelectDataType(const InputOutputInfo& info, ppl::common::datatype_t forward_precision,
                                 std::vector<dataformat_t>* selected_input_data_types,
                                 std::vector<dataformat_t>* selected_output_data_types) {
    const uint32_t input_count = info.GetInputCount();
    if (DATATYPE_INT64 == selected_input_data_types->at(0)) {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_data_types->at(i) = DATATYPE_INT64;
        }
        selected_output_data_types->at(0) = DATATYPE_INT64;
    } else {
        for (uint32_t i = 0; i < input_count; i++) {
            selected_input_data_types->at(i) = forward_precision;
        }
        selected_output_data_types->at(0) = forward_precision;
    }

    return RC_SUCCESS;
}

KernelImpl* ConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConcatKernel>(param_.get());
}

}}}; // namespace ppl::nn::riscv
