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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/concat_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_concat.h"
#include "ppl/nn/common/logger.h"

#include <set>

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ConcatOp::ConcatOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConcat(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

RetCode ConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ConcatOp::SelectFormat(const InputOutputInfo& info,
                               std::vector<ppl::common::dataformat_t>* selected_input_formats,
                               std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    const uint32_t input_count = info.GetInputCount();
    std::set<dataformat_t> input_dataformats;
    for (uint32_t i = 0; i < input_count; i++) {
        input_dataformats.insert(info.GetInput<TensorImpl>(i)->GetShape()->GetDataFormat());
    }

    dataformat_t data_format;
    if (input_dataformats.size() == 1) { // all input has same dataformat
        data_format = *input_dataformats.begin();
    } else { // all data format fall back to NDARRAY
        data_format = DATAFORMAT_NDARRAY;
    }

    for (uint32_t i = 0; i < input_count; i++) {
        selected_input_formats->at(i) = data_format;
    }
    selected_output_formats->at(0) = data_format;

    return RC_SUCCESS;
}

KernelImpl* ConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConcatKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
