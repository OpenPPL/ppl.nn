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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/expand_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/expand_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_expand.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ExpandOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeExpand(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode ExpandOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                               vector<dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
    selected_input_formats->at(1) = DATAFORMAT_NDARRAY;
    selected_output_formats->at(0) = DATAFORMAT_NDARRAY;

    return RC_SUCCESS;
}

RetCode ExpandOp::SelectDataType(const InputOutputInfo& info, datatype_t forward_precision,
                                 std::vector<datatype_t>* selected_input_data_types,
                                 std::vector<datatype_t>* selected_output_data_types) {
    selected_input_data_types->at(1) = DATATYPE_INT64;
    selected_output_data_types->at(0) = selected_input_data_types->at(0);

    return RC_SUCCESS;
}

KernelImpl* ExpandOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ExpandKernel>();
}

}}} // namespace ppl::nn::riscv