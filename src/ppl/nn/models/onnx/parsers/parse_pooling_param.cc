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

#include "ppl/nn/models/onnx/parsers/parse_pooling_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParsePoolingParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::PoolingParam*>(arg);

    param->kernel_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "kernel_shape");
    param->dilations = utils::GetNodeAttrsByKey<int32_t>(pb_node, "dilations");
    param->strides = utils::GetNodeAttrsByKey<int32_t>(pb_node, "strides");
    param->pads = utils::GetNodeAttrsByKey<int32_t>(pb_node, "pads");
    param->ceil_mode = utils::GetNodeAttrByKey<int32_t>(pb_node, "ceil_mode", 0);

    int32_t count_include_pad = utils::GetNodeAttrByKey<int32_t>(pb_node, "count_include_pad", 0);

    std::string auto_pad = utils::GetNodeAttrByKey<std::string>(pb_node, "auto_pad", std::string());
    if (!auto_pad.empty()) {
        LOG(ERROR) << "auto_pad is a DEPRECATED attribute and not supported.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (pb_node.op_type() == "GlobalAveragePool") {
        param->global_pooling = true;
        param->mode = ppl::nn::common::PoolingParam::POOLING_AVERAGE_EXCLUDE;
        return ppl::common::RC_SUCCESS;
    } else {
        param->global_pooling = false;
    }

    if (pb_node.op_type() == "MaxPool") {
        param->mode = ppl::nn::common::PoolingParam::POOLING_MAX;
    } else if (pb_node.op_type() == "AveragePool") {
        if (count_include_pad) {
            param->mode = ppl::nn::common::PoolingParam::POOLING_AVERAGE_INCLUDE;
        } else {
            param->mode = ppl::nn::common::PoolingParam::POOLING_AVERAGE_EXCLUDE;
        }
    } else {
        LOG(ERROR) << "unexpected op type: " << pb_node.op_type();
        return ppl::common::RC_INVALID_VALUE;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
