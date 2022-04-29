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

#include <float.h>

#include "ppl/nn/models/onnx/parsers/onnx/parse_lstm_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseLSTMParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<LSTMParam*>(arg);

    static const map<string, LSTMParam::activation_t> act_map = {
        {"Relu", LSTMParam::ACT_RELU},
        {"Tanh", LSTMParam::ACT_TANH},
        {"Sigmoid", LSTMParam::ACT_SIGMOID},
        {"Affine", LSTMParam::ACT_AFFINE},
        {"LeakyRelu", LSTMParam::ACT_LEAKY_RELU},
        {"ThresholdedRelu", LSTMParam::ACT_THRESHOLDED_RELU},
        {"ScaledTanh", LSTMParam::ACT_SCALED_TANH},
        {"HardSigmoid", LSTMParam::ACT_HARD_SIGMOID},
        {"Elu", LSTMParam::ACT_ELU},
        {"Softsign", LSTMParam::ACT_SOFTSIGN},
        {"Softplus", LSTMParam::ACT_SOFTPLUS},
    };

    static const map<string, LSTMParam::direction_t> direction_map = {
        {"forward", LSTMParam::DIR_FORWARD},
        {"reverse", LSTMParam::DIR_REVERSE},
        {"bidirectional", LSTMParam::DIR_BIDIRECTIONAL},
    };

    param->activation_alpha = utils::GetNodeAttrsByKey<float>(pb_node, "activation_alpha");
    param->activation_beta = utils::GetNodeAttrsByKey<float>(pb_node, "activation_beta");

    auto activations = utils::GetNodeAttrsByKey<string>(pb_node, "activations");
    param->activations.resize(activations.size());
    for (size_t i = 0; i < activations.size(); ++i) {
        auto it = act_map.find(activations[i]);
        if (it == act_map.end()) {
            LOG(ERROR) << "Unsupported activation type: " << activations[i];
        }
        param->activations[i] = it->second;
    }

    param->clip = utils::GetNodeAttrByKey<float>(pb_node, "clip", FLT_MAX);

    auto direction = utils::GetNodeAttrByKey<string>(pb_node, "direction", "forward");
    {
        auto it = direction_map.find(direction);
        if (it == direction_map.end()) {
            LOG(ERROR) << "Unsupported direction type: " << direction;
        }
        param->direction = it->second;
    }

    param->hidden_size = utils::GetNodeAttrByKey<int32_t>(pb_node, "hidden_size", INT32_MIN);
    if (param->hidden_size == INT32_MIN) {
        LOG(ERROR) << "hidden_size is not set but required";
        return RC_INVALID_VALUE;
    }

    param->input_forget = utils::GetNodeAttrByKey<int32_t>(pb_node, "input_forget", 0);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
