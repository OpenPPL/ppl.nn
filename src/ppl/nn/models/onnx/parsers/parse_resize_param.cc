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

#include "ppl/nn/models/onnx/parsers/parse_resize_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseResizeParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ResizeParam*>(arg);

    std::string coordinate_transformation_mode =
        utils::GetNodeAttrByKey<std::string>(pb_node, "coordinate_transformation_mode", std::string());
    if (coordinate_transformation_mode == "") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_HALF_PIXEL;
    } else if (coordinate_transformation_mode == "half_pixel") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_HALF_PIXEL;
    } else if (coordinate_transformation_mode == "pytorch_half_pixel") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL;
    } else if (coordinate_transformation_mode == "align_corners") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_ALIGN_CORNERS;
    } else if (coordinate_transformation_mode == "asymmetric") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_ASYMMETRIC;
    } else if (coordinate_transformation_mode == "tf_half_pixel_for_nn") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN;
    } else if (coordinate_transformation_mode == "tf_crop_and_resize") {
        param->coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_TF_CROP_AND_RESIZE;
    } else {
        LOG(ERROR) << "unexpected coordinate_transformation_mode: " << coordinate_transformation_mode;
        return ppl::common::RC_INVALID_VALUE;
    }

    param->cubic_coeff_a = utils::GetNodeAttrByKey(pb_node, "cubic_coeff_a", -0.75f);
    param->exclude_outside = utils::GetNodeAttrByKey(pb_node, "exclude_outside", 0);
    param->extrapolation_value = utils::GetNodeAttrByKey(pb_node, "extrapolation_value", 0.0f);

    std::string mode = utils::GetNodeAttrByKey<std::string>(pb_node, "mode", std::string());
    if (mode == "") {
        param->mode = ppl::nn::common::ResizeParam::RESIZE_MODE_NEAREST;
    } else if (mode == "nearest") {
        param->mode = ppl::nn::common::ResizeParam::RESIZE_MODE_NEAREST;
    } else if (mode == "linear") {
        param->mode = ppl::nn::common::ResizeParam::RESIZE_MODE_LINEAR;
    } else if (mode == "cubic") {
        param->mode = ppl::nn::common::ResizeParam::RESIZE_MODE_CUBIC;
    } else {
        LOG(ERROR) << "unexpected mode: " << mode;
        return ppl::common::RC_INVALID_VALUE;
    }

    std::string nearest_mode = utils::GetNodeAttrByKey<std::string>(pb_node, "nearest_mode", std::string());
    if (nearest_mode == "") {
        param->nearest_mode = ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR;
    } else if (nearest_mode == "round_prefer_floor") {
        param->nearest_mode = ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR;
    } else if (nearest_mode == "round_prefer_ceil") {
        param->nearest_mode = ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL;
    } else if (nearest_mode == "floor") {
        param->nearest_mode = ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_FLOOR;
    } else if (nearest_mode == "ceil") {
        param->nearest_mode = ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_CEIL;
    } else {
        LOG(ERROR) << "unexpected nearest_mode: " << nearest_mode;
        return ppl::common::RC_INVALID_VALUE;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
