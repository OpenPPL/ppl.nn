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

#include "ppl/nn/models/onnx/parsers/onnx/parse_conv_param.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseConvParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<ConvParam*>(arg);

    auto auto_pad_str = utils::GetNodeAttrByKey<string>(pb_node, "auto_pad", "NOSET");
    if (auto_pad_str == "NOSET") {
        param->auto_pad = ConvParam::NOSET;
    } else if (auto_pad_str == "SAME_UPPER") {
        param->auto_pad = ConvParam::SAME_UPPER;
    } else if (auto_pad_str == "SAME_LOWER") {
        param->auto_pad = ConvParam::SAME_LOWER;
    } else if (auto_pad_str == "VALID") {
        param->auto_pad = ConvParam::VALID;
    } else {
        LOG(ERROR) << "unsupported auto_pad type: " << auto_pad_str;
        return RC_UNSUPPORTED;
    }

    param->group = utils::GetNodeAttrByKey(pb_node, "group", 1);
    param->kernel_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "kernel_shape");
    param->dilations = utils::GetNodeAttrsByKey<int32_t>(pb_node, "dilations");
    param->strides = utils::GetNodeAttrsByKey<int32_t>(pb_node, "strides");
    param->pads = utils::GetNodeAttrsByKey<int32_t>(pb_node, "pads");

    uint32_t kernel_dims = param->kernel_shape.size();
    if (kernel_dims == 0) {
        return RC_INVALID_VALUE;
    }

    // if empty, set to default value
    if (param->dilations.size() == 0) {
        param->dilations.resize(kernel_dims, 1);
    }
    if (param->strides.size() == 0) {
        param->strides.resize(kernel_dims, 1);
    }
    if (param->pads.size() == 0) {
        param->pads.resize(kernel_dims * 2, 0);
    }

    if (param->dilations.size() != kernel_dims || param->strides.size() != kernel_dims ||
        param->pads.size() != kernel_dims * 2) {
        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
