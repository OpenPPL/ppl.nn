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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/pad_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/pad_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pad.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

RetCode PadOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto ret = onnx::ReshapePad(info, param_.get());
        if (ret != RC_SUCCESS) {
            return ret;
        }

        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode PadOp::SelectDataType(const InputOutputInfo& info, std::vector<ppl::common::datatype_t>* selected_input_types,
                              std::vector<ppl::common::datatype_t>* selected_output_types,
                              const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    selected_input_types->at(1) = ppl::common::DATATYPE_INT64;
    return RC_SUCCESS;
}

RetCode PadOp::SelectFormat(const InputOutputInfo& info, std::vector<ppl::common::dataformat_t>* selected_input_formats,
                            std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    const auto input_format = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    auto selected_dataformat = input_format;
    if (input_format == ppl::common::DATAFORMAT_N4CX ||
        input_format ==
            ppl::common::DATAFORMAT_N8CX) { // for nbcx pad, if pad on channel dim, will fall back to ndarray implement
        const auto pads_data = info.GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
        if (pads_data == nullptr) { // pads not sure on compiler time, fall back to ndarray implement
            selected_dataformat = ppl::common::DATAFORMAT_NDARRAY;
        } else {
            const auto start_pads = pads_data;
            const auto end_pads = pads_data + info.GetInput<TensorImpl>(0)->GetShape()->GetDimCount();
            const int64_t c_dim_idx = 1;
            if (start_pads[c_dim_idx] != 0 || end_pads[c_dim_idx] != 0) {
                selected_dataformat = ppl::common::DATAFORMAT_NDARRAY;
            }
        }
    }

    selected_input_formats->at(0) = selected_output_formats->at(0) = selected_dataformat;
    for (uint32_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_formats->at(i) = ppl::common::DATAFORMAT_NDARRAY;
    }
    return RC_SUCCESS;
}

KernelImpl* PadOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PadKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
