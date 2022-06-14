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

#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_UTILS_H
#define _ST_HPC_PPL_NN_MODELS_ONNX_UTILS_H

#include "ppl/nn/models/onnx/generated/onnx.pb.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/common/types.h"

namespace ppl { namespace nn { namespace onnx { namespace utils {

template <typename T>
struct IntValueSetter final {
    IntValueSetter(::onnx::NodeProto* pb_node, const std::string& key, const T& value) {
        auto pb_attr = pb_node->add_attribute();
        pb_attr->set_name(key);
        pb_attr->set_i(value);
        pb_attr->set_type(::onnx::AttributeProto_AttributeType_INT);
    }
};

template <typename T>
struct FloatValueSetter final {
    FloatValueSetter(::onnx::NodeProto* pb_node, const std::string& key, const T& value) {
        auto pb_attr = pb_node->add_attribute();
        pb_attr->set_name(key);
        pb_attr->set_f(value);
        pb_attr->set_type(::onnx::AttributeProto_AttributeType_FLOAT);
    }
};

template <typename T>
void SetNodeAttr(::onnx::NodeProto* pb_node, const std::string& key, const T& value) {
    typename std::conditional<std::is_integral<T>::value, IntValueSetter<T>,
                              typename std::conditional<std::is_floating_point<T>::value, FloatValueSetter<T>,
                                                        void>::type>::type setter(pb_node, key, value);
}

template <typename T>
struct IntVecSetter final {
    IntVecSetter(::onnx::NodeProto* pb_node, const std::string& key, const std::vector<T>& values) {
        if (values.size() > 0) {
            auto pb_attr = pb_node->add_attribute();
            pb_attr->set_name(key);
            for (uint32_t i = 0; i < values.size(); ++i) {
                pb_attr->add_ints(values[i]);
            }
            pb_attr->set_type(::onnx::AttributeProto_AttributeType_INTS);
        }
    }
};

template <typename T>
struct FloatVecSetter final {
    FloatVecSetter(::onnx::NodeProto* pb_node, const std::string& key, const std::vector<T>& values) {
        if (values.size() > 0) {
            auto pb_attr = pb_node->add_attribute();
            pb_attr->set_name(key);
            for (uint32_t i = 0; i < values.size(); ++i) {
                pb_attr->add_floats(values[i]);
            }
            pb_attr->set_type(::onnx::AttributeProto_AttributeType_FLOATS);
        }
    }
};

template <typename T>
void SetNodeAttr(::onnx::NodeProto* pb_node, const std::string& key, const std::vector<T>& values) {
    typename std::conditional<std::is_integral<T>::value, IntVecSetter<T>,
                              typename std::conditional<std::is_floating_point<T>::value, FloatVecSetter<T>,
                                                        void>::type>::type setter(pb_node, key, values);
}

inline void SetNodeAttr(::onnx::NodeProto* pb_node, const std::string& key, const char* value) {
    auto pb_attr = pb_node->add_attribute();
    pb_attr->set_name(key);
    pb_attr->set_s(value);
    pb_attr->set_type(::onnx::AttributeProto_AttributeType_STRING);
}

inline void SetNodeAttr(::onnx::NodeProto* pb_node, const std::string& key, const std::string& value) {
    auto pb_attr = pb_node->add_attribute();
    pb_attr->set_name(key);
    pb_attr->set_s(value);
    pb_attr->set_type(::onnx::AttributeProto_AttributeType_STRING);
}

inline void SetNodeAttr(::onnx::NodeProto* pb_node, const std::string& key, const std::vector<std::string>& values) {
    auto pb_attr = pb_node->add_attribute();
    pb_attr->set_name(key);
    for (uint32_t i = 0; i < values.size(); ++i) {
        pb_attr->add_strings(values[i]);
    }
    pb_attr->set_type(::onnx::AttributeProto_AttributeType_STRINGS);
}

/* -------------------------------------------------------------------------- */

template <typename T>
struct IntValueGetter final {
    IntValueGetter(const ::onnx::AttributeProto& pb_attr, T* value) {
        *value = pb_attr.i();
    }
};

template <typename T>
struct FloatValueGetter final {
    FloatValueGetter(const ::onnx::AttributeProto& pb_attr, T* value) {
        *value = pb_attr.f();
    }
};

template <typename T, typename TDefault>
inline void GetNodeAttr(const ::onnx::NodeProto& pb_node, const char* key, T* value, TDefault default_value) {
    for (int i = 0; i < pb_node.attribute_size(); ++i) {
        auto& pb_attr = pb_node.attribute(i);
        if (pb_attr.name() == key) {
            typename std::conditional<std::is_integral<T>::value, IntValueGetter<T>,
                                      typename std::conditional<std::is_floating_point<T>::value, FloatValueGetter<T>,
                                                                void>::type>::type getter(pb_attr, value);
            return;
        }
    }
    *value = default_value;
}

template <typename T>
struct IntVecGetter final {
    IntVecGetter(const ::onnx::AttributeProto& pb_attr, std::vector<T>* values) {
        values->resize(pb_attr.ints_size());
        for (int j = 0; j < pb_attr.ints_size(); ++j) {
            values->at(j) = pb_attr.ints(j);
        }
    }
};

template <typename T>
struct FloatVecGetter final {
    FloatVecGetter(const ::onnx::AttributeProto& pb_attr, std::vector<T>* values) {
        values->resize(pb_attr.floats_size());
        for (int j = 0; j < pb_attr.floats_size(); ++j) {
            values->at(j) = pb_attr.floats(j);
        }
    }
};

template <typename T>
inline void GetNodeAttr(const ::onnx::NodeProto& pb_node, const char* key, std::vector<T>* values) {
    for (int i = 0; i < pb_node.attribute_size(); ++i) {
        auto& pb_attr = pb_node.attribute(i);
        if (pb_attr.name() == key) {
            typename std::conditional<std::is_integral<T>::value, IntVecGetter<T>,
                                      typename std::conditional<std::is_floating_point<T>::value, FloatVecGetter<T>,
                                                                void>::type>::type getter(pb_attr, values);
            return;
        }
    }
}

inline void GetNodeAttr(const ::onnx::NodeProto& pb_node, const char* key, std::vector<std::string>* values) {
    for (int i = 0; i < pb_node.attribute_size(); ++i) {
        auto& pb_attr = pb_node.attribute(i);
        if (pb_attr.name() == key) {
            values->resize(pb_attr.strings_size());
            for (int j = 0; i < pb_attr.strings_size(); ++i) {
                values->at(j) = pb_attr.strings(j);
            }
            return;
        }
    }
}

inline void GetNodeAttr(const ::onnx::NodeProto& pb_node, const char* key, std::string* value,
                        const std::string& default_value) {
    for (int i = 0; i < pb_node.attribute_size(); ++i) {
        auto& pb_attr = pb_node.attribute(i);
        if (pb_attr.name() == key) {
            *value = pb_attr.s();
            return;
        }
    }
    *value = default_value;
}

inline void GetNodeAttr(const ::onnx::NodeProto& pb_node, const char* key, std::string* value,
                        const char* default_value) {
    GetNodeAttr(pb_node, key, value, std::string(default_value));
}

/* -------------------------------------------------------------------------- */

const ::onnx::TensorProto* GetTensorProtoByKey(const ::onnx::NodeProto&, const char* key);

ppl::common::RetCode ParseTensorProto(const ::onnx::TensorProto&, const char* model_file_dir, std::string*, ir::Shape*);
ppl::common::RetCode PackTensorProto(const void* data, uint64_t len, ppl::common::datatype_t,
                                     const std::vector<int64_t>&, ::onnx::TensorProto*);

/* -------------------------------------------------------------------------- */

ppl::common::datatype_t ConvertOnnxDataTypeToPplDataType(int32_t data_type);
int32_t ConvertPplDataTypeToOnnxDataType(ppl::common::datatype_t);

/* -------------------------------------------------------------------------- */

void ResolveExtraInputs(ir::GraphTopo* current, ir::Node* parent_node, ir::GraphTopo* parent_graph);

}}}} // namespace ppl::nn::onnx::utils

#endif
