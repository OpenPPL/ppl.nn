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

#include "ppl/nn/models/onnx/parsers/opmx/parse_pixel_unshuffle_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::opmx;

namespace ppl { namespace nn { namespace opmx {

RetCode ParsePixelUnshuffleParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node,
                                  ir::Attr* arg) {
    auto param = static_cast<PixelUnshuffleParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "scale_factor", &param->scale_factor, -1)) {
        LOG(ERROR) << node->GetName() << ": missing scale_factor";
        return RC_INVALID_VALUE;
    }

    string data_layout;
    if (!onnx::utils::GetNodeAttr(pb_node, "data_layout", &data_layout, "")) {
        LOG(ERROR) << node->GetName() << ": missing data_layout";
        return RC_INVALID_VALUE;
    }

    if (data_layout == "nhwc") {
        param->data_layout = ppl::nn::opmx::PixelUnshuffleParam::DATA_LAYOUT_NHWC;
    } else {
        LOG(ERROR) << "unsupported data_layout: " << data_layout;
        return RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx
