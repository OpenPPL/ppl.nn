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

#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/lstm.h"
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx { namespace onnx {

Offset<LSTMParam> SerializeLSTMParam(const ppl::nn::onnx::LSTMParam& param, FlatBufferBuilder* builder) {
    auto fb_activation_alpha = builder->CreateVector(param.activation_alpha);
    auto fb_activation_beta = builder->CreateVector(param.activation_beta);
    std::vector<uint32_t> temp_activations(param.activations.size());
    for (uint32_t i = 0; i < param.activations.size(); i++) {
        temp_activations[i] = static_cast<uint32_t>(param.activations[i]);
    }
    auto fb_activations = builder->CreateVector(temp_activations);
    return CreateLSTMParam(*builder, fb_activation_alpha, fb_activation_beta, fb_activations, param.clip,
                           static_cast<LSTMDirectionType>(param.direction), param.hidden_size, param.input_forget);
}

void DeserializeLSTMParam(const LSTMParam& fb_param, ppl::nn::onnx::LSTMParam* param) {
    utils::Fbvec2Stdvec(fb_param.activation_alpha(), &param->activation_alpha);
    utils::Fbvec2Stdvec(fb_param.activation_beta(), &param->activation_beta);
    param->activations.resize(fb_param.activations()->size());
    for (uint32_t i = 0; i < fb_param.activations()->size(); i++) {
        param->activations.at(i) = fb_param.activations()->Get(i);
    }
    param->clip = fb_param.clip();
    param->direction = fb_param.direction();
    param->hidden_size = fb_param.hidden_size();
    param->input_forget = fb_param.input_forget();
}

}}}} // namespace ppl::nn::pmx::onnx
