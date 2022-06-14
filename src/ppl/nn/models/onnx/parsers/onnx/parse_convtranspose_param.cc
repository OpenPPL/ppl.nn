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
#include "ppl/nn/models/onnx/parsers/onnx/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseConvTransposeParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*,
                                ir::Attr* arg) {
    auto param = static_cast<ConvTransposeParam*>(arg);

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
        LOG(ERROR) << "unsupported auto_pad type: " << auto_pad_str;
        return RC_UNSUPPORTED;
    }

    utils::GetNodeAttr(pb_node, "dilations", &param->dilations);
    utils::GetNodeAttr(pb_node, "kernel_shape", &param->kernel_shape);
    utils::GetNodeAttr(pb_node, "output_padding", &param->output_padding);
    utils::GetNodeAttr(pb_node, "output_shape", &param->output_shape);
    utils::GetNodeAttr(pb_node, "pads", &param->pads);
    utils::GetNodeAttr(pb_node, "strides", &param->strides);
    utils::GetNodeAttr(pb_node, "group", &param->group, 1);

    return RC_SUCCESS;
}

RetCode PackConvTransposeParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const ConvTransposeParam*>(arg);

    if (param->auto_pad == AUTO_PAD_NOTSET) {
        utils::SetNodeAttr(pb_node, "auto_pad", "NOTSET");
    } else if (param->auto_pad == AUTO_PAD_SAME_UPPER) {
        utils::SetNodeAttr(pb_node, "auto_pad", "SAME_UPPER");
    } else if (param->auto_pad == AUTO_PAD_SAME_LOWER) {
        utils::SetNodeAttr(pb_node, "auto_pad", "SAME_LOWER");
    } else if (param->auto_pad == AUTO_PAD_VALID) {
        utils::SetNodeAttr(pb_node, "auto_pad", "VALID");
    } else {
        LOG(ERROR) << "unsupported auto_pad type: " << param->auto_pad;
        return RC_UNSUPPORTED;
    }

    utils::SetNodeAttr(pb_node, "dilations", param->dilations);
    utils::SetNodeAttr(pb_node, "group", param->group);
    utils::SetNodeAttr(pb_node, "kernel_shape", param->kernel_shape);
    utils::SetNodeAttr(pb_node, "output_padding", param->output_padding);
    utils::SetNodeAttr(pb_node, "output_shape", param->output_shape);
    utils::SetNodeAttr(pb_node, "pads", param->pads);
    utils::SetNodeAttr(pb_node, "strides", param->strides);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
