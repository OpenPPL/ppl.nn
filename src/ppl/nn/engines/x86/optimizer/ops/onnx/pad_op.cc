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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/pad_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/pad_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pad.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode PadOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        uint32_t dim_count = info->GetInput<TensorImpl>(0)->GetShape().GetDimCount();
        auto pad = info->GetInput<TensorImpl>(1);
        if (pad->GetShape().GetDimCount() != 1 || pad->GetShape().GetDim(0) != 2 * dim_count ||
            pad->GetShape().GetDataType() != DATATYPE_INT64) {
            return RC_INVALID_VALUE;
        }

        auto ret = oputils::ReshapePad(info, param_.get());
        if (ret != RC_SUCCESS) {
            return ret;
        }

        if (info->GetInput<TensorImpl>(0)->GetShape().GetDataType() != DATATYPE_FLOAT32) {
            LOG(ERROR) << "unsupported data type: "
                       << GetDataTypeStr(info->GetInput<TensorImpl>(0)->GetShape().GetDataType());
            return RC_UNSUPPORTED;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode PadOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                            vector<dataformat_t>* selected_output_formats) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}

KernelImpl* PadOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PadKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
