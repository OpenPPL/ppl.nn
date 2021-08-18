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

#include "ppl/nn/models/onnx/parsers/parse_mmcv_modulated_deform_conv2d_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseMMCVModulatedDeformConv2dParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::MMCVModulatedDeformConv2dParam*>(arg);

    param->kernel_size[0] = 0; // set by opcontext, for converted filter
    param->kernel_size[1] = 0; // set by opcontext, for converted filter
    param->channels = 0; // set by opcontext, for converted filter
    param->num_output = 0; // set by opcontext, for converted filter
    param->bias_term = 0; // set by opcontext, for multi-input layer fusion

    param->groups = utils::GetNodeAttrByKey(pb_node, "groups", 1);
    param->deform_groups = utils::GetNodeAttrByKey(pb_node, "deform_groups", 1);

    auto stride = utils::GetNodeAttrsByKey<int32_t>(pb_node, "stride");
    auto padding = utils::GetNodeAttrsByKey<int32_t>(pb_node, "padding");
    auto dilation = utils::GetNodeAttrsByKey<int32_t>(pb_node, "dilation");

    auto kernel_dims = 2;

    if (dilation.size() != kernel_dims || stride.size() != kernel_dims || padding.size() != kernel_dims) {
        return ppl::common::RC_INVALID_VALUE;
    }

    param->stride[0] = stride[0];
    param->stride[1] = stride[1];

    param->padding[0] = padding[0];
    param->padding[1] = padding[1];

    param->dilation[0] = dilation[0];
    param->dilation[1] = dilation[1];

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
