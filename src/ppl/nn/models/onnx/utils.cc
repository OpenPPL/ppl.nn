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

#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx { namespace utils {

template <>
vector<int32_t> GetNodeAttrsByKey<int32_t>(const ::onnx::NodeProto& node, const char* key) {
    vector<int32_t> result;
    for (int32_t i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            result.resize(attribute.ints_size());
            for (int32_t j = 0; j < attribute.ints_size(); j++) {
                result[j] = attribute.ints(j);
            }
            break;
        }
    }
    return result;
}

template <>
vector<float> GetNodeAttrsByKey<float>(const ::onnx::NodeProto& node, const char* key) {
    vector<float> result;
    for (int32_t i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            result.resize(attribute.floats_size());
            for (int32_t j = 0; j < attribute.floats_size(); j++) {
                result[j] = attribute.floats(j);
            }
            break;
        }
    }
    return result;
}

template <>
vector<string> GetNodeAttrsByKey<string>(const ::onnx::NodeProto& node, const char* key) {
    vector<string> result;
    for (int32_t i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            result.resize(attribute.strings_size());
            for (int32_t j = 0; j < attribute.strings_size(); j++) {
                result[j] = attribute.strings(j);
            }
            break;
        }
    }
    return result;
}

template <>
int32_t GetAttrValue<int32_t>(const ::onnx::AttributeProto& attribute) {
    return attribute.i();
}

template <>
int64_t GetAttrValue<int64_t>(const ::onnx::AttributeProto& attribute) {
    return attribute.i();
}

template <>
string GetAttrValue<string>(const ::onnx::AttributeProto& attribute) {
    return attribute.s();
}

template <>
float GetAttrValue<float>(const ::onnx::AttributeProto& attribute) {
    return attribute.f();
}

const ::onnx::TensorProto* GetTensorProtoByKey(const ::onnx::NodeProto& node, const char* key) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            return &attribute.t();
        }
    }
    return nullptr;
}

datatype_t ConvertOnnxDataTypeToPplDataType(int32_t onnx_data_type) {
    static datatype_t dt_map[] = {
        DATATYPE_UNKNOWN, // unknown
        DATATYPE_FLOAT32, // float32
        DATATYPE_UINT8, // uint8
        DATATYPE_INT8, // int8
        DATATYPE_UINT16, // uint16
        DATATYPE_INT16, // int16
        DATATYPE_INT32, // int32
        DATATYPE_INT64, // int64
        DATATYPE_UNKNOWN, // String
        DATATYPE_BOOL, // bool
        DATATYPE_FLOAT16, // float16
        DATATYPE_FLOAT64, // float64
        DATATYPE_UINT32, // uint32
        DATATYPE_UINT64, // uint64
        DATATYPE_COMPLEX64, // complex64
        DATATYPE_COMPLEX128, // complex128
        DATATYPE_BFLOAT16, // bfloat16
    };
    static const int32_t onnx_data_type_max = sizeof(dt_map) / sizeof(datatype_t);

    if (onnx_data_type >= onnx_data_type_max) {
        return DATATYPE_UNKNOWN;
    }

    return dt_map[onnx_data_type];
}

RetCode ParseTensorProto(const ::onnx::TensorProto& pb_tensor, string* data, ir::Shape* shape) {
    const int32_t onnx_data_type = pb_tensor.data_type();
    const datatype_t ppl_data_type = utils::ConvertOnnxDataTypeToPplDataType(onnx_data_type);
    const uint32_t elem_size = GetSizeOfDataType(ppl_data_type);

    shape->data_type = ppl_data_type;
    shape->data_format = DATAFORMAT_NDARRAY; // default data format
    for (int j = 0; j < pb_tensor.dims_size(); ++j) {
        auto dim = pb_tensor.dims(j);
        shape->dims.push_back(dim);
    }

    if (onnx_data_type == ::onnx::TensorProto_DataType_FLOAT) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.float_data_size() > 0) {
            data->assign((const char*)pb_tensor.float_data().data(), pb_tensor.float_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_DOUBLE) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.double_data_size() > 0) {
            data->assign((const char*)pb_tensor.double_data().data(), pb_tensor.double_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_INT32) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.int32_data_size() > 0) {
            data->assign((const char*)pb_tensor.int32_data().data(), pb_tensor.int32_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_INT64) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.int64_data().size() > 0) {
            data->assign((const char*)pb_tensor.int64_data().data(), pb_tensor.int64_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_BOOL) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.int32_data().size() > 0) { // bool may be stored in int32_data
            data->assign((const char*)pb_tensor.int32_data().data(), pb_tensor.int32_data().size() * sizeof(int32_t));
        }
    } else {
        auto onnx_pb_type = (::onnx::TensorProto_DataType)onnx_data_type;
        LOG(ERROR) << "unsupported onnx data type[" << ::onnx::TensorProto_DataType_Name(onnx_pb_type) << "] of tensor["
                   << pb_tensor.name() << "]";
        return RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

void ResolveExtraInputs(ir::GraphTopo* current, ir::Node* parent_node, ir::GraphTopo* parent_graph) {
    for (uint32_t i = 0; i < current->GetExtraInputCount(); ++i) {
        auto edge = current->GetEdgeById(current->GetExtraInput(i));

        auto ret_pair = parent_graph->AddEdge(edge->GetName());
        auto edge_of_parent = ret_pair.first;

        if (ret_pair.second) {
            parent_graph->MarkAsExtraInput(edge_of_parent->GetId());
        }

        edge_of_parent->AddConsumer(parent_node->GetId());
        parent_node->AddExtraInput(edge_of_parent->GetId());
    }
}

}}}} // namespace ppl::nn::onnx::utils
