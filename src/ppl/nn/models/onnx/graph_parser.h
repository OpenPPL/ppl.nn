#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_GRAPH_PARSER_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_GRAPH_PARSER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/models/onnx/generated/onnx.pb.h"

namespace ppl { namespace nn { namespace onnx {

class GraphParser final {
public:
    ppl::common::RetCode Parse(const ::onnx::GraphProto& pb_graph, ir::Graph* graph);

private:
    uint32_t anonymous_node_count_ = 0; // used to generate anonymous node name
};

}}} // namespace ppl::nn::onnx

#endif
