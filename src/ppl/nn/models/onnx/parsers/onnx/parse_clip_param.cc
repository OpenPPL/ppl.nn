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

#include "ppl/nn/models/onnx/parsers/onnx/parse_clip_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseClipParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<ClipParam*>(arg);
    param->min_value = utils::GetNodeAttrByKey<float>(pb_node, "min", numeric_limits<float>::lowest());
    param->max_value = utils::GetNodeAttrByKey<float>(pb_node, "max", numeric_limits<float>::max());
    return RC_SUCCESS;
}

RetCode PackClipParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const ClipParam*>(arg);
    utils::SetNodeAttr(pb_node, "min", param->min_value);
    utils::SetNodeAttr(pb_node, "max", param->max_value);
    return RC_SUCCESS;
}

RetCode PackClipParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const ClipParam*>(arg);
    utils::SetNodeAttr(pb_node, "min", param->min_value);
    utils::SetNodeAttr(pb_node, "max", param->max_value);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
