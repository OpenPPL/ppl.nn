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

#include "ppl/nn/models/onnx/parsers/parse_constant_of_shape_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseConstantOfShapeParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConstantOfShapeParam*>(arg);

    const ::onnx::TensorProto* value = utils::GetTensorProtoByKey(pb_node, "value");
    if (value == nullptr) {
        float f = 0.0;
        param->data_type = ppl::common::DATATYPE_FLOAT32;
        param->dims.push_back(1);
        param->data.assign((const char*)&f, sizeof(f));
    } else {
        ir::Shape shape;
        auto status = utils::ParseTensorProto(*value, &param->data, &shape);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "parse attribute of node[" << pb_node.name()
                       << "] failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }

        param->data_type = shape.data_type;
        param->dims = std::move(shape.dims);
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
