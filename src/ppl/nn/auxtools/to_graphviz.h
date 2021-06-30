#ifndef _ST_HPC_PPL_NN_AUXTOOLS_TO_GRAPHVIZ_H_
#define _ST_HPC_PPL_NN_AUXTOOLS_TO_GRAPHVIZ_H_

#include <string>
#include "ppl/nn/ir/graph_topo.h"
using namespace std;

namespace ppl { namespace nn { namespace utils {

static string ToGraphviz(const ir::GraphTopo* topo) {
    string content = "digraph NetGraph {\n";

    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();

        string begin_node_name;
        if (edge->GetProducer() == INVALID_NODEID) {
            begin_node_name = "NIL-BEGIN";
        } else {
            auto node = topo->GetNodeById(edge->GetProducer());
            begin_node_name = node->GetName();
        }

        auto edge_iter = edge->CreateConsumerIter();
        if (edge_iter.IsValid()) {
            do {
                auto node = topo->GetNodeById(edge_iter.Get());
                content +=
                    "\"" + begin_node_name + "\" -> \"" + node->GetName() + "\" [label=\"" + edge->GetName() + "\"]\n";
                edge_iter.Forward();
            } while (edge_iter.IsValid());
        } else {
            content += "\"" + begin_node_name + "\" -> \"NIL-END\" [label=\"" + edge->GetName() + "\"]\n";
            edge_iter.Forward();
        }
    }

    content += "}";
    return content;
}

}}} // namespace ppl::nn::utils

#endif
