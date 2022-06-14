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

#include "ppl/nn/models/onnx/parsers/onnx/parse_hard_sigmoid_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseHardSigmoidParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*,
                              ir::Attr* arg) {
    auto param = static_cast<HardSigmoidParam*>(arg);
    utils::GetNodeAttr(pb_node, "alpha", &param->alpha, 0.2f);
    utils::GetNodeAttr(pb_node, "beta", &param->beta, 0.5f);
    return RC_SUCCESS;
}

RetCode PackHardSigmoidParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const HardSigmoidParam*>(arg);
    utils::SetNodeAttr(pb_node, "alpha", param->alpha);
    utils::SetNodeAttr(pb_node, "beta", param->beta);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
