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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/conv_transpose_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/conv_transpose_kernel.h"
#include "ppl/nn/engines/riscv/utils/fp16fp32_cvt.h"
#include "ppl/nn/oputils/onnx/reshape_convtranspose.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

template <typename T>
static void CvtGraphData(const float* graph_data, T* cvt_data, int32_t data_len) {
    auto data_bytes = data_len * sizeof(T);
    if (typeid(T) == typeid(__fp16)) {
        CvtFp32ToFp16(data_len, graph_data, cvt_data);
    } else if (typeid(T) == typeid(float)) {
        memcpy(cvt_data, graph_data, data_bytes);
    } else {
        memcpy(cvt_data, graph_data, data_bytes);
    }
}

RetCode ConvTransposeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConvTranspose(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
    conv_transpose_param_ = shared_ptr<ppl::nn::riscv::ConvTransposeParam>(new ppl::nn::riscv::ConvTransposeParam());
    conv_transpose_param_->param = param_;

    if (options.engine_options->forward_precision == DATATYPE_FLOAT32) {
        conv_transpose_param_->algo_info =
            ppl::kernel::riscv::conv_transpose_fp32_algo_selector::select_algo(options.engine_options);
    } else if (options.engine_options->forward_precision == DATATYPE_FLOAT16) {
        conv_transpose_param_->algo_info =
            ppl::kernel::riscv::conv_transpose_fp16_algo_selector::select_algo(options.engine_options);
    } else {
        return RC_UNSUPPORTED;
    }

    auto node = GetNode();
    auto graph_data = options.graph_data;
    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    const float* weight_data = nullptr;
    if (weight_data_it != graph_data->constants.end()) {
        weight_data = (const float*)weight_data_it->second.data.GetData();
        int64_t weight_len = weight_data_it->second.data.GetSize() / sizeof(float);

        if (options.engine_options->forward_precision == DATATYPE_FLOAT32) {
            shared_ptr<void> weight_cvt(malloc(weight_len * sizeof(float)), free);
            CvtGraphData<float>(weight_data, (float*)weight_cvt.get(), weight_len);
            conv_transpose_param_->weight = weight_cvt;
        } else if (options.engine_options->forward_precision == DATATYPE_FLOAT16) {
            shared_ptr<void> weight_cvt(malloc(weight_len * sizeof(__fp16)), free);
            CvtGraphData<__fp16>(weight_data, (__fp16*)weight_cvt.get(), weight_len);
            conv_transpose_param_->weight = weight_cvt;
        } else {
            return RC_UNSUPPORTED;
        }
    }

    const float* bias_data = nullptr;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it != graph_data->constants.end()) {
            bias_data = (const float*)bias_data_it->second.data.GetData();
            int64_t bias_len = bias_data_it->second.data.GetSize() / sizeof(float);
            if (options.engine_options->forward_precision == DATATYPE_FLOAT32) {
                shared_ptr<void> bias_cvt(malloc(bias_len * sizeof(float)), free);
                CvtGraphData<float>(bias_data, (float*)bias_cvt.get(), bias_len);
                conv_transpose_param_->bias = bias_cvt;
            } else if (options.engine_options->forward_precision == DATATYPE_FLOAT16) {
                shared_ptr<void> bias_cvt(malloc(bias_len * sizeof(__fp16)), free);
                CvtGraphData<__fp16>(bias_data, (__fp16*)bias_cvt.get(), bias_len);
                conv_transpose_param_->bias = bias_cvt;
            } else {
                return RC_UNSUPPORTED;
            }
        }
    }

    return RC_SUCCESS;
}

RetCode ConvTransposeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                      vector<dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = conv_transpose_param_->algo_info.input_format;
    selected_output_formats->at(0) = conv_transpose_param_->algo_info.output_format;
    return RC_SUCCESS;
}

RetCode ConvTransposeOp::SelectDataType(const InputOutputInfo& info, ppl::common::datatype_t forward_precision,
                                        std::vector<dataformat_t>* selected_input_data_types,
                                        std::vector<dataformat_t>* selected_output_data_types) {
    selected_input_data_types->at(0) = conv_transpose_param_->algo_info.input_data_type;
    selected_output_data_types->at(0) = conv_transpose_param_->algo_info.output_data_type;
    return RC_SUCCESS;
}

KernelImpl* ConvTransposeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConvTransposeKernel>(conv_transpose_param_.get());
}

}}} // namespace ppl::nn::riscv
