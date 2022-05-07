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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/shape_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ShapeOp::ShapeOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->Reshape({info->GetInput<TensorImpl>(0)->GetShape()->GetRealDimCount()});
        return RC_SUCCESS;
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataType(DATATYPE_INT64);
    };
}

RetCode ShapeOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode ShapeOp::SelectDataType(const InputOutputInfo& info, std::vector<ppl::common::datatype_t>* selected_input_types,
                                std::vector<ppl::common::datatype_t>* selected_output_types,
                                const ppl::common::datatype_t preferred_fp_datatype) {
    selected_input_types->at(0) = info.GetInput<TensorImpl>(0)->GetShape()->GetDataType();
    selected_output_types->at(0) = DATATYPE_INT64;
    return RC_SUCCESS;
}

RetCode ShapeOp::SelectFormat(const InputOutputInfo& info,
                              std::vector<ppl::common::dataformat_t>* selected_input_formats,
                              std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    return RC_SUCCESS;
}

KernelImpl* ShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ShapeKernel>();
}

}}} // namespace ppl::nn::arm
