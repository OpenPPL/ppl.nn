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

#include "ppl/nn/engines/riscv/optimizer/ops/pmx/shape_operation_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/common/pmx/shape_operation_kernel.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ShapeOperationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::ShapeOperationParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    infer_type_func_ = GenericInferType;
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = [](InputOutputInfo* info) -> void {
        GenericInferType(info);
        info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(DATATYPE_INT64);
    };

    return RC_SUCCESS;
}

RetCode ShapeOperationOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    auto num_output = selected_output_formats->size();
    for (uint64_t i = 0; i < num_output; i += 1) {
        selected_output_formats->at(i) = DATAFORMAT_NDARRAY;
    }
    return RC_SUCCESS;
}

RetCode ShapeOperationOp::SelectDataType(const InputOutputInfo& info, ppl::common::datatype_t forward_precision,
                                         std::vector<datatype_t>* selected_input_data_types,
                                         std::vector<datatype_t>* selected_output_data_types) {
    auto num_output = selected_output_data_types->size();
    for (uint64_t i = 0; i < num_output; i += 1) {
        selected_output_data_types->at(i) = DATATYPE_INT64;
    }
    return RC_SUCCESS;
}

KernelImpl* ShapeOperationOp::CreateKernelImpl() const {
    auto kernel = op_.CreateKernelImpl();
    ((ppl::nn::pmx::ShapeOperationKernel*)kernel)->SetParam(param_.get());
    return kernel;
}

}}} // namespace ppl::nn::riscv
