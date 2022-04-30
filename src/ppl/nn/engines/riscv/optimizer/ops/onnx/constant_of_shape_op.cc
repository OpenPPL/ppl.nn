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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_constant_of_shape.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ConstantOfShapeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConstantOfShape(info, param_.get());
    };

    infer_type_func_ = [this](InputOutputInfo* info) -> void {
        TensorShape* shape = info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(param_->data_type);
    };

    return RC_SUCCESS;
}

RetCode ConstantOfShapeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                        vector<dataformat_t>* selected_output_formats) {
    selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    return RC_SUCCESS;
}

RetCode ConstantOfShapeOp::SelectDataType(const InputOutputInfo& info, ppl::common::datatype_t forward_precision,
                                          std::vector<dataformat_t>* selected_input_data_types,
                                          std::vector<dataformat_t>* selected_output_data_types) {
    selected_output_data_types->at(0) = param_->data_type;
    return RC_SUCCESS;
}

KernelImpl* ConstantOfShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConstantOfShapeKernel>(param_.get());
}

}}} // namespace ppl::nn::riscv