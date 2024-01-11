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

#include "parse_linear_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace pmx {

RetCode ParseLinearParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node, ir::Attr* arg) {

    auto param = static_cast<LinearParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "in_features", &param->in_features, -1)) {
        LOG(ERROR) << node->GetName() << ": missing in_features";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "out_features", &param->out_features, -1)) {
        LOG(ERROR) << node->GetName() << ": missing out_features";
        return RC_INVALID_VALUE;
    }

    int32_t bias_term;
    onnx::utils::GetNodeAttr(pb_node, "bias_term", &bias_term, 1);
    param->bias_term = bias_term;

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
