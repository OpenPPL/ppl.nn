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

#include "ppl/nn/models/onnx/parsers/onnx/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseConvTransposeParam(const ::onnx::NodeProto& pb_node, const map<string, uint64_t>&, void* arg,
                                             ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConvTransposeParam*>(arg);

    param->auto_pad = utils::GetNodeAttrByKey<string>(pb_node, "auto_pad", "");
    param->dilations = utils::GetNodeAttrsByKey<int32_t>(pb_node, "dilations");
    param->kernel_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "kernel_shape");
    param->output_padding = utils::GetNodeAttrsByKey<int32_t>(pb_node, "output_padding");
    param->output_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "output_shape");
    param->pads = utils::GetNodeAttrsByKey<int32_t>(pb_node, "pads");
    param->strides = utils::GetNodeAttrsByKey<int32_t>(pb_node, "strides");
    param->group = utils::GetNodeAttrByKey<int64_t>(pb_node, "group", 1);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
