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

#include "ppl/nn/models/onnx/parsers/onnx/parse_topk_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace onnx {
RetCode ParseTopKParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto it = args.op_set->find(pb_node.domain());
    if (it == args.op_set->end()) {
        return RC_INVALID_VALUE;
    }
    auto opset = it->second;

    auto param = static_cast<TopKParam*>(arg);

    param->axis = utils::GetNodeAttrByKey<int32_t>(pb_node, "axis", -1);
    param->largest = utils::GetNodeAttrByKey<int32_t>(pb_node, "largest", 1);
    param->sorted = utils::GetNodeAttrByKey<int32_t>(pb_node, "sorted", 1);
    param->k = utils::GetNodeAttrByKey<int32_t>(pb_node, "k", -1);

    if (opset < 10 && param->k == -1) {
        return RC_NOT_FOUND;
    }

    return RC_SUCCESS;
}
}}} // namespace ppl::nn::onnx
