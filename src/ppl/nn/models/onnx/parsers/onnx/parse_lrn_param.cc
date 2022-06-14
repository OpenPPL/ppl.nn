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

#include "ppl/nn/models/onnx/parsers/onnx/parse_lrn_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseLRNParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<LRNParam*>(arg);

    utils::GetNodeAttr(pb_node, "size", &param->size, INT32_MAX);
    if (param->size == INT32_MAX) {
        LOG(ERROR) << "size is required.";
        return RC_INVALID_VALUE;
    }

    utils::GetNodeAttr(pb_node, "alpha", &param->alpha, 0.0001f);
    utils::GetNodeAttr(pb_node, "beta", &param->beta, 0.75f);
    utils::GetNodeAttr(pb_node, "bias", &param->bias, 1.0f);
    return RC_SUCCESS;
}

RetCode PackLRNParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const LRNParam*>(arg);
    utils::SetNodeAttr(pb_node, "alpha", param->alpha);
    utils::SetNodeAttr(pb_node, "beta", param->beta);
    utils::SetNodeAttr(pb_node, "bias", param->bias);
    utils::SetNodeAttr(pb_node, "size", param->size);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
