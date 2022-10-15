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

#include "ppl/nn/models/onnx/parsers/onnx/parse_gru_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static const vector<pair<string, int32_t>> g_activation_values = {
    {"Relu", GRUParam::ACT_RELU},
    {"Tanh", GRUParam::ACT_TANH},
    {"Sigmoid", GRUParam::ACT_SIGMOID},
    {"Affine", GRUParam::ACT_AFFINE},
    {"LeakyRelu", GRUParam::ACT_LEAKY_RELU},
    {"ThresholdedRelu", GRUParam::ACT_THRESHOLDED_RELU},
    {"ScaledTanh", GRUParam::ACT_SCALED_TANH},
    {"HardSigmoid", GRUParam::ACT_HARD_SIGMOID},
    {"Elu", GRUParam::ACT_ELU},
    {"Softsign", GRUParam::ACT_SOFTSIGN},
    {"Softplus", GRUParam::ACT_SOFTPLUS},
};

static const vector<pair<string, int32_t>> g_direction_values = {
    {"forward", GRUParam::DIR_FORWARD},
    {"reverse", GRUParam::DIR_REVERSE},
    {"bidirectional", GRUParam::DIR_BIDIRECTIONAL},
};

RetCode ParseGRUParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<GRUParam*>(arg);

    utils::GetNodeAttr(pb_node, "activation_alpha", &param->activation_alpha);
    utils::GetNodeAttr(pb_node, "activation_beta", &param->activation_beta);

    vector<string> activations;
    utils::GetNodeAttr(pb_node, "activations", &activations);

    param->activations.resize(activations.size());
    for (size_t i = 0; i < activations.size(); ++i) {
        auto& act_str = activations[i];
        auto it = std::find_if(g_activation_values.begin(), g_activation_values.end(),
                               [&act_str](const pair<string, int32_t>& p) -> bool {
                                   return (act_str == p.first);
                               });
        if (it == g_activation_values.end()) {
            LOG(ERROR) << "unsupported activation type: " << activations[i];
            return RC_UNSUPPORTED;
        }
        param->activations[i] = it->second;
    }

    utils::GetNodeAttr(pb_node, "clip", &param->clip, FLT_MAX);

    string direction;
    utils::GetNodeAttr(pb_node, "direction", &direction, "forward");
    {
        auto it = std::find_if(g_direction_values.begin(), g_direction_values.end(),
                               [&direction](const pair<string, int32_t>& p) -> bool {
                                   return (direction == p.first);
                               });
        if (it == g_direction_values.end()) {
            LOG(ERROR) << "unsupported direction type: " << direction;
            return RC_UNSUPPORTED;
        }
        param->direction = it->second;
    }

    utils::GetNodeAttr(pb_node, "hidden_size", &param->hidden_size, INT32_MIN);
    if (param->hidden_size == INT32_MIN) {
        LOG(ERROR) << "hidden_size is not set but required";
        return RC_INVALID_VALUE;
    }

    utils::GetNodeAttr(pb_node, "linear_before_reset", &param->linear_before_reset, 1); // linear_before_reset = 1 in pytorch

    return RC_SUCCESS;
}

RetCode PackGRUParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const GRUParam*>(arg);

    utils::SetNodeAttr(pb_node, "activation_alpha", param->activation_alpha);
    utils::SetNodeAttr(pb_node, "activation_beta", param->activation_beta);
    utils::SetNodeAttr(pb_node, "clip", param->clip);
    utils::SetNodeAttr(pb_node, "hidden_size", param->hidden_size);
    utils::SetNodeAttr(pb_node, "linear_before_reset", param->linear_before_reset);

    auto it = std::find_if(g_direction_values.begin(), g_direction_values.end(),
                           [param](const pair<string, int32_t>& p) -> bool {
                               return (param->direction == p.second);
                           });
    if (it == g_direction_values.end()) {
        LOG(ERROR) << "unsupported direction type: " << param->direction;
        return RC_UNSUPPORTED;
    }
    utils::SetNodeAttr(pb_node, "direction", it->first);

    vector<string> act_strs;
    for (auto x = param->activations.begin(); x != param->activations.end(); ++x) {
        auto act_value = *x;
        it = std::find_if(g_activation_values.begin(), g_activation_values.end(),
                          [&act_value](const pair<string, int32_t>& p) -> bool {
                              return (act_value == p.second);
                          });
        if (it == g_activation_values.end()) {
            LOG(ERROR) << "unsupported activation type: " << act_value;
            return RC_UNSUPPORTED;
        }
        act_strs.push_back(it->first);
    }
    utils::SetNodeAttr(pb_node, "activations", act_strs);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
