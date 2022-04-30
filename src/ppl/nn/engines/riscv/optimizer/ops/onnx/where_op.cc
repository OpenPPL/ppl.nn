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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/where_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_where.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode WhereOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeWhere(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        GenericInferType(info);
        info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(
            info->GetInput<TensorImpl>(1)->GetShape()->GetDataType());
    };

    return RC_SUCCESS;
}

RetCode WhereOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                              vector<dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
    selected_input_formats->at(1) = DATAFORMAT_NDARRAY;
    selected_input_formats->at(2) = DATAFORMAT_NDARRAY;
    selected_output_formats->at(0) = DATAFORMAT_NDARRAY;

    return RC_SUCCESS;
}

RetCode WhereOp::SelectDataType(const InputOutputInfo& info, ppl::common::datatype_t forward_precision,
                                std::vector<datatype_t>* selected_input_data_types,
                                std::vector<datatype_t>* selected_output_data_types) {
    selected_input_data_types->at(0) = DATATYPE_BOOL;
    if (DATATYPE_FLOAT16 == selected_input_data_types->at(1)) {
        selected_input_data_types->at(2) = DATATYPE_FLOAT16;
        selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    } else if (DATATYPE_FLOAT32 == selected_input_data_types->at(1)) {
        selected_input_data_types->at(2) = DATATYPE_FLOAT32;
        selected_output_data_types->at(0) = DATATYPE_FLOAT32;
    } else if (DATATYPE_INT64 == selected_input_data_types->at(1)) {
        selected_input_data_types->at(2) = DATATYPE_INT64;
        selected_output_data_types->at(0) = DATATYPE_INT64;
    }

    return RC_SUCCESS;
}

KernelImpl* WhereOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<WhereKernel>();
}

}}} // namespace ppl::nn::riscv