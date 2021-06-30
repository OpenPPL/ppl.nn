#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_UTILS_H
#define _ST_HPC_PPL_NN_MODELS_ONNX_UTILS_H

#include "ppl/nn/models/onnx/generated/onnx.pb.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/common/types.h"

namespace ppl { namespace nn { namespace onnx { namespace utils {

template <typename T>
std::vector<T> GetNodeAttrsByKey(const ::onnx::NodeProto& node, const char* key);

template <typename T>
T GetAttrValue(const ::onnx::AttributeProto& attribute);

template <typename T>
T GetNodeAttrByKey(const ::onnx::NodeProto& node, const char* key, T default_value) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            return GetAttrValue<T>(attribute);
        }
    }
    return default_value;
}

const ::onnx::TensorProto* GetTensorProtoByKey(const ::onnx::NodeProto&, const char* key);

ppl::common::RetCode ParseTensorProto(const ::onnx::TensorProto&, std::string*, ir::Shape*);

ppl::common::datatype_t ConvertOnnxDataTypeToPplDataType(int32_t data_type);

void ResolveExtraInputs(ir::GraphTopo* current, ir::Node* parent_node, ir::GraphTopo* parent_graph);

}}}} // namespace ppl::nn::onnx::utils

#endif
