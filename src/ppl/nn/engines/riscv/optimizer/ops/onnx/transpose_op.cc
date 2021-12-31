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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/transpose_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_transpose.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode TransposeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeTranspose(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode TransposeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                  vector<dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
    selected_output_formats->at(0) = DATAFORMAT_NDARRAY;

    return RC_SUCCESS;
}

RetCode TransposeOp::SelectDataType(const InputOutputInfo& info, vector<dataformat_t>* selected_input_data_types,
                                    vector<dataformat_t>* selected_output_data_types) {
    if (DATATYPE_FLOAT16 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    } else if (DATATYPE_FLOAT32 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT32;
    }

    return RC_SUCCESS;
}

KernelImpl* TransposeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<TransposeKernel>(param_.get());
}

}}} // namespace ppl::nn::riscv
