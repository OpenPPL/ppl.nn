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

#include "ppl/nn/models/onnx/parsers/onnx/parse_random_uniform_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseRandomUniformParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<RandomUniformParam*>(arg);
    utils::GetNodeAttr(pb_node, "dtype", &param->dtype, 1);
    param->dtype = utils::ConvertOnnxDataTypeToPplDataType(param->dtype);
    utils::GetNodeAttr(pb_node, "high", &param->high, 1.0f);
    utils::GetNodeAttr(pb_node, "low", &param->low, 0.0f);
    utils::GetNodeAttr(pb_node, "shape", &param->shape);
    
    // if there is no seed, the vector will stay empty
    float seed;
    bool has_seed = utils::GetNodeAttr(pb_node, "seed", &seed, 0.0f);
    if (has_seed) {
        param->seed.assign({seed});
    }
    return RC_SUCCESS;
}

RetCode PackRandomUniformParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const RandomUniformParam*>(arg);
    utils::SetNodeAttr(pb_node, "dtype", param->dtype);
    utils::SetNodeAttr(pb_node, "high", param->high);
    utils::SetNodeAttr(pb_node, "low", param->low);
    utils::SetNodeAttr(pb_node, "shape", param->shape);
    if (param->seed.size()) {
        utils::SetNodeAttr(pb_node, "seed", param->seed[0]);
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
