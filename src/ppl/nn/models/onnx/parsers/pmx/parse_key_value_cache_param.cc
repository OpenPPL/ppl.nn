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

#include "ppl/nn/models/onnx/parsers/pmx/parse_key_value_cache_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace pmx {

RetCode ParseKeyValueCacheParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node,
                                 ir::Attr* arg) {
    auto param = static_cast<KeyValueCacheParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "num_layer", &param->num_layer, -1)) {
        LOG(ERROR) << node->GetName() << ": missing num_layer";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "layer_idx", &param->layer_idx, -1)) {
        LOG(ERROR) << node->GetName() << ": missing layer_idx";
        return RC_INVALID_VALUE;
    }

    onnx::utils::GetNodeAttr(pb_node, "quant_bit", &param->quant_bit, 0);
    onnx::utils::GetNodeAttr(pb_node, "quant_group", &param->quant_group, 8);
    onnx::utils::GetNodeAttr(pb_node, "num_repeat", &param->num_repeat, 1);
    onnx::utils::GetNodeAttr(pb_node, "cache_mode", &param->cache_mode, 0);
    onnx::utils::GetNodeAttr(pb_node, "cache_layout", &param->cache_layout, 0);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
