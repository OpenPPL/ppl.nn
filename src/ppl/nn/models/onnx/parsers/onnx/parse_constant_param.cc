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

#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static RetCode DoParseTensorProto(const ::onnx::TensorProto& pb_tensor, ppl::nn::common::ConstantParam* param) {
    ir::Shape shape;
    auto status = utils::ParseTensorProto(pb_tensor, &param->data, &shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse `value` failed: " << GetRetCodeStr(status);
        return status;
    }

    param->data_type = shape.data_type;
    param->data_format = shape.data_format;
    param->dims = std::move(shape.dims);

    return RC_SUCCESS;
}

static const unordered_set<string> g_unsupported_fields = {"sparse_value", "value_string", "value_strings"};

RetCode ParseConstantParam(const ::onnx::NodeProto& pb_node, const map<string, uint64_t>&, void* arg, ir::Node*,
                           ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConstantParam*>(arg);

    for (int i = 0; i < pb_node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = pb_node.attribute(i);
        if (g_unsupported_fields.find(attribute.name()) != g_unsupported_fields.end()) {
            LOG(ERROR) << "attribute[" << attribute.name() << "] is not supported in current implementation.";
            return RC_UNSUPPORTED;
        }

        if (attribute.name() == "value") {
            return DoParseTensorProto(attribute.t(), param);
        } else if (attribute.name() == "value_int") {
            int64_t v = attribute.i();
            param->data_type = DATATYPE_INT64;
            param->data_format = DATAFORMAT_NDARRAY;
            param->dims.push_back(1);
            param->data.assign((const char*)(&v), sizeof(v));
            return RC_SUCCESS;
        } else if (attribute.name() == "value_ints") {
            param->data_type = DATATYPE_INT64;
            param->data_format = DATAFORMAT_NDARRAY;
            param->dims.push_back(attribute.ints_size());
            param->data.reserve(attribute.ints_size() * sizeof(int64_t));
            for (int x = 0; x < attribute.ints_size(); ++x) {
                int64_t v = attribute.ints(x);
                param->data.append((const char*)(&v), sizeof(v));
            }
            return RC_SUCCESS;
        } else if (attribute.name() == "value_float") {
            float v = attribute.f();
            param->data_type = DATATYPE_FLOAT32;
            param->data_format = DATAFORMAT_NDARRAY;
            param->dims.push_back(1);
            param->data.assign((const char*)(&v), sizeof(v));
            return RC_SUCCESS;
        } else if (attribute.name() == "value_floats") {
            param->data_type = DATATYPE_FLOAT32;
            param->data_format = DATAFORMAT_NDARRAY;
            param->dims.push_back(attribute.floats_size());
            param->data.reserve(attribute.floats_size() * sizeof(float));
            for (int x = 0; x < attribute.floats_size(); ++x) {
                float v = attribute.floats(x);
                param->data.append((const char*)(&v), sizeof(v));
            }
            return RC_SUCCESS;
        }
    }

    LOG(ERROR) << "cannot find supported fields in Constant[" << pb_node.name() << "]";
    return RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::onnx
