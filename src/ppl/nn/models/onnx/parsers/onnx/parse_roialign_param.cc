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

#include "ppl/nn/models/onnx/parsers/onnx/parse_roialign_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseRoiAlignParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*,
                           ir::Attr* arg) {
    auto param = static_cast<RoiAlignParam*>(arg);

    string mode;
    utils::GetNodeAttr(pb_node, "mode", &mode, "avg");
    if (mode == "avg") {
        param->mode = RoiAlignParam::AVG;
    } else if (mode == "max") {
        param->mode = RoiAlignParam::MAX;
    } else {
        LOG(ERROR) << "Invalid mode " << mode << ".";
        return RC_INVALID_VALUE;
    }
    utils::GetNodeAttr(pb_node, "output_height", &param->output_height, 1);
    utils::GetNodeAttr(pb_node, "output_width", &param->output_width, 1);
    utils::GetNodeAttr(pb_node, "sampling_ratio", &param->sampling_ratio, 0);
    utils::GetNodeAttr(pb_node, "spatial_scale", &param->spatial_scale, 1.0f);
    return RC_SUCCESS;
}

RetCode PackRoiAlignParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const RoiAlignParam*>(arg);

    if (param->mode == RoiAlignParam::MAX) {
        utils::SetNodeAttr(pb_node, "mode", "max");
    } else if (param->mode == RoiAlignParam::AVG) {
        utils::SetNodeAttr(pb_node, "mode", "avg");
    } else {
        LOG(ERROR) << "invalid mode[" << param->mode << "]";
        return RC_INVALID_VALUE;
    }

    utils::SetNodeAttr(pb_node, "output_height", param->output_height);
    utils::SetNodeAttr(pb_node, "output_width", param->output_width);
    utils::SetNodeAttr(pb_node, "sampling_ratio", param->sampling_ratio);
    utils::SetNodeAttr(pb_node, "spatial_scale", param->spatial_scale);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
