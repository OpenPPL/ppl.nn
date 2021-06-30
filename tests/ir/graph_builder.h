#ifndef _ST_HPC_PPL_NN_TESTS_IR_GRAPH_BUILDER_H_
#define _ST_HPC_PPL_NN_TESTS_IR_GRAPH_BUILDER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"
#include <string>
#include <vector>

namespace ppl { namespace nn { namespace test {

class GraphBuilder final {
public:
    GraphBuilder();
    ppl::common::RetCode AddNode(const std::string& name, const ir::Node::Type& type,
                                 const std::vector<std::string>& inputs, const std::vector<std::string>& outputs);
    ppl::common::RetCode Finalize();
    ir::Graph* GetGraph() const {
        return &graph_;
    }
    void SetGraphName(const std::string& name) {
        graph_.topo->SetName(name);
    }

private:
    mutable ir::Graph graph_;
};

}}} // namespace ppl::nn::test

#endif
