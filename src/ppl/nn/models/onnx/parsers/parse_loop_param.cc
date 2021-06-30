#include "ppl/nn/models/onnx/parsers/parse_loop_param.h"
#include "ppl/nn/models/onnx/graph_parser.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseLoopParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node* node, ir::GraphTopo* topo) {
    auto param = static_cast<LoopParam*>(arg);
    if (pb_node.attribute_size() != 1) {
        LOG(ERROR) << "invalid attribute size[" << pb_node.attribute_size() << "] != 1.";
        return RC_INVALID_VALUE;
    }
    auto& attr = pb_node.attribute(0);
    if (attr.type() != ::onnx::AttributeProto_AttributeType_GRAPH) {
        LOG(ERROR) << "unsupported attribute type[" << ::onnx::AttributeProto_AttributeType_Name(attr.type()) << "]";
        return RC_INVALID_VALUE;
    }

    GraphParser parser;
    auto status = parser.Parse(attr.g(), &(param->graph));
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse subgraph of loop pb_node[" << pb_node.name();
        return status;
    }

    utils::ResolveExtraInputs(param->graph.topo.get(), node, topo);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
