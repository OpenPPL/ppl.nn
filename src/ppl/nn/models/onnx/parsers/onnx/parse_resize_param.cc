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

#include "ppl/nn/models/onnx/parsers/onnx/parse_resize_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static const vector<pair<string, int32_t>> g_coord_trans_modes = {
    {"half_pixel", ResizeParam::RESIZE_COORD_TRANS_MODE_HALF_PIXEL},
    {"pytorch_half_pixel", ResizeParam::RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL},
    {"align_corners", ResizeParam::RESIZE_COORD_TRANS_MODE_ALIGN_CORNERS},
    {"asymmetric", ResizeParam::RESIZE_COORD_TRANS_MODE_ASYMMETRIC},
    {"tf_half_pixel_for_nn", ResizeParam::RESIZE_COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN},
    {"tf_crop_and_resize", ResizeParam::RESIZE_COORD_TRANS_MODE_TF_CROP_AND_RESIZE},
};

static const vector<pair<string, int32_t>> g_modes = {
    {"nearest", ResizeParam::RESIZE_MODE_NEAREST},
    {"linear", ResizeParam::RESIZE_MODE_LINEAR},
    {"cubic", ResizeParam::RESIZE_MODE_CUBIC},
};

static const vector<pair<string, int32_t>> g_nearest_modes = {
    {"round_prefer_floor", ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR},
    {"round_prefer_ceil", ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL},
    {"floor", ResizeParam::RESIZE_NEAREST_MODE_FLOOR},
    {"ceil", ResizeParam::RESIZE_NEAREST_MODE_CEIL},
};

RetCode ParseResizeParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<ResizeParam*>(arg);

    string str;
    utils::GetNodeAttr(pb_node, "coordinate_transformation_mode", &str, "half_pixel");
    auto it = std::find_if(g_coord_trans_modes.begin(), g_coord_trans_modes.end(),
                           [&str](const pair<string, int32_t>& p) -> bool {
                               return (str == p.first);
                           });
    if (it == g_coord_trans_modes.end()) {
        LOG(ERROR) << "unexpected coordinate_transformation_mode: " << str;
        return RC_INVALID_VALUE;
    }
    param->coord_trans_mode = it->second;

    utils::GetNodeAttr(pb_node, "cubic_coeff_a", &param->cubic_coeff_a, -0.75f);
    utils::GetNodeAttr(pb_node, "exclude_outside", &param->exclude_outside, 0);
    utils::GetNodeAttr(pb_node, "extrapolation_value", &param->extrapolation_value, 0.0f);

    utils::GetNodeAttr(pb_node, "mode", &str, "nearest");
    it = std::find_if(g_modes.begin(), g_modes.end(), [&str](const pair<string, int32_t>& p) -> bool {
        return (str == p.first);
    });
    if (it == g_modes.end()) {
        LOG(ERROR) << "unexpected mode: " << str;
        return RC_INVALID_VALUE;
    }
    param->mode = it->second;

    utils::GetNodeAttr(pb_node, "nearest_mode", &str, "round_prefer_floor");
    it = std::find_if(g_nearest_modes.begin(), g_nearest_modes.end(),
                      [&str](const pair<string, int32_t>& p) -> bool {
                          return (str == p.first);
                      });
    if (it == g_nearest_modes.end()) {
        LOG(ERROR) << "unexpected nearest_mode: " << param->nearest_mode;
        return RC_INVALID_VALUE;
    }
    param->nearest_mode = it->second;

    return RC_SUCCESS;
}

RetCode PackResizeParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const ResizeParam*>(arg);

    auto it = std::find_if(g_coord_trans_modes.begin(), g_coord_trans_modes.end(),
                           [param](const pair<string, int32_t>& p) -> bool {
                               return (param->coord_trans_mode == p.second);
                           });
    if (it == g_coord_trans_modes.end()) {
        LOG(ERROR) << "unexpected coord trans mode[" << param->coord_trans_mode << "]";
        return RC_INVALID_VALUE;
    }
    utils::SetNodeAttr(pb_node, "coordinate_transformation_mode", it->first);

    utils::SetNodeAttr(pb_node, "cubic_coeff_a", param->cubic_coeff_a);
    utils::SetNodeAttr(pb_node, "exclude_outside", param->exclude_outside);
    utils::SetNodeAttr(pb_node, "extrapolation_value", param->extrapolation_value);

    it = std::find_if(g_modes.begin(), g_modes.end(), [param](const pair<string, int32_t>& p) -> bool {
        return (param->mode == p.second);
    });
    if (it == g_modes.end()) {
        LOG(ERROR) << "unexpected mode[" << param->mode << "]";
        return RC_INVALID_VALUE;
    }
    utils::SetNodeAttr(pb_node, "mode", it->first);

    it = std::find_if(g_nearest_modes.begin(), g_nearest_modes.end(), [param](const pair<string, int32_t>& p) -> bool {
        return (param->nearest_mode == p.second);
    });
    if (it == g_nearest_modes.end()) {
        LOG(ERROR) << "unexpected nearest mode[" << param->nearest_mode << "]";
        return RC_INVALID_VALUE;
    }
    utils::SetNodeAttr(pb_node, "nearest_mode", it->first);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
