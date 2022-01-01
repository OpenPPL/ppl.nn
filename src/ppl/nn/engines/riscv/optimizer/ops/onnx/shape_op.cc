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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/shape_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ShapeOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->Reshape({info->GetInput<TensorImpl>(0)->GetShape()->GetRealDimCount()});
        return RC_SUCCESS;
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

RetCode ShapeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                              vector<dataformat_t>* selected_output_formats) {
    selected_output_formats->at(0) = ppl::common::DATAFORMAT_NDARRAY;
    return RC_SUCCESS;
}

RetCode ShapeOp::SelectDataType(const InputOutputInfo& info, vector<datatype_t>* selected_input_data_types,
                                vector<datatype_t>* selected_output_data_types) {
    selected_output_data_types->at(0) = DATATYPE_INT64;
    return RC_SUCCESS;
}

KernelImpl* ShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ShapeKernel>();
}

}}} // namespace ppl::nn::riscv
