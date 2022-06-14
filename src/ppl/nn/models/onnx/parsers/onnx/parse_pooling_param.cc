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

#include "ppl/nn/params/onnx/auto_pad_type.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pooling_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParsePoolingParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*,
                          ir::Attr* arg) {
    auto param = static_cast<PoolingParam*>(arg);

    utils::GetNodeAttr(pb_node, "ceil_mode", &param->ceil_mode, 0);
    utils::GetNodeAttr(pb_node, "dilations", &param->dilations);
    utils::GetNodeAttr(pb_node, "kernel_shape", &param->kernel_shape);
    utils::GetNodeAttr(pb_node, "pads", &param->pads);
    utils::GetNodeAttr(pb_node, "storage_order", &param->storage_order, 0);
    utils::GetNodeAttr(pb_node, "strides", &param->strides);

    string auto_pad_str;
    utils::GetNodeAttr(pb_node, "auto_pad", &auto_pad_str, "NOTSET");
    if (auto_pad_str == "NOTSET") {
        param->auto_pad = AUTO_PAD_NOTSET;
    } else if (auto_pad_str == "SAME_UPPER") {
        param->auto_pad = AUTO_PAD_SAME_UPPER;
    } else if (auto_pad_str == "SAME_LOWER") {
        param->auto_pad = AUTO_PAD_SAME_LOWER;
    } else if (auto_pad_str == "VALID") {
        param->auto_pad = AUTO_PAD_VALID;
    } else {
        LOG(ERROR) << "unsupported auto_pad type [" << auto_pad_str << "]";
        return RC_UNSUPPORTED;
    }

    if (pb_node.op_type() == "GlobalAveragePool") {
        param->global_pooling = true;
        param->mode = PoolingParam::POOLING_AVERAGE_EXCLUDE;
        return RC_SUCCESS;
    } else {
        param->global_pooling = false;
    }

    if (pb_node.op_type() == "MaxPool") {
        param->mode = PoolingParam::POOLING_MAX;
    } else if (pb_node.op_type() == "AveragePool") {
        int32_t count_include_pad;
        utils::GetNodeAttr(pb_node, "count_include_pad", &count_include_pad, 0);
        if (count_include_pad) {
            param->mode = PoolingParam::POOLING_AVERAGE_INCLUDE;
        } else {
            param->mode = PoolingParam::POOLING_AVERAGE_EXCLUDE;
        }
    } else {
        LOG(ERROR) << "unexpected op type: " << pb_node.op_type();
        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}

RetCode PackPoolingParam(const ir::Node* node, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto& node_type = node->GetType();
    if (node_type.name == "GlobalAveragePool" || node_type.name == "GlobalMaxPool") {
        return RC_SUCCESS;
    }

    auto param = static_cast<const PoolingParam*>(arg);

    if (param->auto_pad == AUTO_PAD_NOTSET) {
        utils::SetNodeAttr(pb_node, "auto_pad", "NOTSET");
    } else if (param->auto_pad == AUTO_PAD_SAME_UPPER) {
        utils::SetNodeAttr(pb_node, "auto_pad", "SAME_UPPER");
    } else if (param->auto_pad == AUTO_PAD_SAME_LOWER) {
        utils::SetNodeAttr(pb_node, "auto_pad", "SAME_LOWER");
    } else if (param->auto_pad == AUTO_PAD_VALID) {
        utils::SetNodeAttr(pb_node, "auto_pad", "VALID");
    } else {
        LOG(ERROR) << "unsupported auto pad type[" << param->auto_pad << "]";
        return RC_UNSUPPORTED;
    }

    if (node_type.version > 9) {
        utils::SetNodeAttr(pb_node, "ceil_mode", param->ceil_mode);
        utils::SetNodeAttr(pb_node, "dilations", param->dilations);
    }
    utils::SetNodeAttr(pb_node, "kernel_shape", param->kernel_shape);
    utils::SetNodeAttr(pb_node, "pads", param->pads);
    utils::SetNodeAttr(pb_node, "strides", param->strides);

    if (node_type.name == "MaxPool") {
        utils::SetNodeAttr(pb_node, "storage_order", param->storage_order);
    } else if (node_type.name == "AveragePool") {
        utils::SetNodeAttr(pb_node, "count_include_pad",
                           (param->mode == PoolingParam::POOLING_AVERAGE_INCLUDE) ? 1 : 0);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
