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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_einsum.h"

#include <chrono>

#include "ppl/common/cuda/cuda_types.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

void EinSumAlgorithm::DeleteAttrParam(void*& param) {
    delete (CudaEinSumParam*)param;
    return;
}

void EinSumAlgorithm::GetAttrParam(void*& param) const {
    if (param == nullptr) {
        param = new CudaEinSumParam();
    }
    *(CudaEinSumParam*)param = attr_param_;
    return;
}

bool EinSumAlgorithm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                dataformat_t input_format) const {
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    if (tensor0.GetDataType() != DATATYPE_FLOAT16 && tensor0.GetDataType() != DATATYPE_INT8) {
        return false;
    }
    if (input_format != DATAFORMAT_NHWC16 && input_format != DATAFORMAT_NHWC8) {
        return false;
    }
    // check if conv is quantization
    if (tensor0.GetDataType() == DATATYPE_INT8 && input_format != DATAFORMAT_NHWC16) {
        return false;
    }
    if (tensor0.GetDataType() == DATATYPE_FLOAT16 && input_format != DATAFORMAT_NHWC8) {
        return false;
    }
    return true;
}

double EinSumAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaEinSumParam*>(options.param));
    options.compile_set->emplace(node->GetId());
    auto shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();

    const std::string& key_str = node->GetName();
    auto algo_info = options.args->alog_selects.find(key_str);
    if (algo_info != options.args->alog_selects.end()) {
        attr_param_.extra_param.algo_info.kid = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
        attr_param_.extra_param.algo_info.algo_name = algo_info->second.kname;
        if (algo_info->second.splitk > 1)
            attr_param_.extra_param.algo_info.algo_name += "_spk" + std::to_string(algo_info->second.splitk);
        attr_param_.extra_param.algo_info.ParseAlgoName();
        return 0.0f;
    } else { // Give the default kernel
        if (shape_in0.GetDataType() == DATATYPE_FLOAT16) {
            attr_param_.extra_param.algo_info.algo_name = "nv2spkSm75Fp16Conv_hmma1688_nhwc_f1_b128x128_w64x64_k32_s32_buf1";
        } else if (shape_in0.GetDataType() == DATATYPE_INT8) {
            attr_param_.extra_param.algo_info.algo_name = "nv2spkSm75Int8Conv_imma8816_nhwc_f1_b64x64_w64x32_k32_s16_buf1";
        } else {
            return ALGO_MAX_TIME;
        }
        attr_param_.extra_param.algo_info.kid = 0; // TODO
        attr_param_.extra_param.algo_info.splitk = 1;
        attr_param_.extra_param.algo_info.splitf = 1;
        attr_param_.extra_param.algo_info.ParseAlgoName();
    }

    return 0.0f;
}

RetCode EinSumAlgorithm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    return RC_SUCCESS;
}

void EinSumAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                   dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1) {
            shape->SetDataFormat(input_format);
        } else {
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
