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

#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_modulated_deform_conv2d_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::mmcv;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseMMCVModulatedDeformConv2dParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args,
                                            ir::Node*, ir::Attr* arg) {
    auto param = static_cast<MMCVModulatedDeformConv2dParam*>(arg);

    utils::GetNodeAttr(pb_node, "groups", &param->groups, 1);
    utils::GetNodeAttr(pb_node, "deform_groups", &param->deform_groups, 1);

    vector<int64_t> stride;
    utils::GetNodeAttr(pb_node, "stride", &stride);

    vector<int64_t> padding;
    utils::GetNodeAttr(pb_node, "padding", &padding);

    vector<int64_t> dilation;
    utils::GetNodeAttr(pb_node, "dilation", &dilation);

    constexpr size_t kernel_dims = 2;
    if (dilation.size() != kernel_dims || stride.size() != kernel_dims || padding.size() != kernel_dims) {
        return RC_INVALID_VALUE;
    }

    param->stride[0] = stride[0];
    param->stride[1] = stride[1];

    param->padding[0] = padding[0];
    param->padding[1] = padding[1];

    param->dilation[0] = dilation[0];
    param->dilation[1] = dilation[1];

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
