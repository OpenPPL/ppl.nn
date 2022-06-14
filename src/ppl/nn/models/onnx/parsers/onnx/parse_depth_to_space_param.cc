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

#include "ppl/nn/models/onnx/parsers/onnx/parse_depth_to_space_param.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseDepthToSpaceParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*,
                               ir::Attr* arg) {
    auto param = static_cast<DepthToSpaceParam*>(arg);

    string mode;
    utils::GetNodeAttr(pb_node, "mode", &mode, "DCR");
    if (mode == "DCR") {
        param->mode = DepthToSpaceParam::DCR;
    } else if (mode == "CRD") {
        param->mode = DepthToSpaceParam::CRD;
    } else {
        LOG(ERROR) << "invalid DepthToSpace mode `" << mode << "`.";
        return RC_INVALID_VALUE;
    }

    utils::GetNodeAttr(pb_node, "blocksize", &param->blocksize, 0);

    return RC_SUCCESS;
}

RetCode PackDepthToSpaceParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const DepthToSpaceParam*>(arg);

    if (param->mode == DepthToSpaceParam::DCR) {
        utils::SetNodeAttr(pb_node, "mode", "DCR");
    } else if (param->mode == DepthToSpaceParam::CRD) {
        utils::SetNodeAttr(pb_node, "mode", "CRD");
    } else {
        LOG(ERROR) << "invalid DepthToSpace mode `" << param->mode << "`.";
        return RC_INVALID_VALUE;
    }

    utils::SetNodeAttr(pb_node, "blocksize", param->blocksize);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
