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

#include "ppl/nn/ir/partial_graph_topo.h"
#include "ppl/nn/auxtools/to_graphviz.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class GraphTopoTest : public testing::Test {
protected:
    void SetUp() override {
        builder_.AddNode("f", ir::Node::Type("test", "op3", 1), {"out2", "out4", "out5"}, {"out6"});
        builder_.AddNode("e", ir::Node::Type("test", "op3", 1), {"out1"}, {"out5"});
        builder_.AddNode("d", ir::Node::Type("test", "op2", 1), {"out3"}, {"out4"});
        builder_.AddNode("c", ir::Node::Type("test", "op2", 1), {"out1"}, {"out3"});
        builder_.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"});
        builder_.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
        builder_.Finalize();
    }

    GraphBuilder builder_;
};

TEST_F(GraphTopoTest, ReplaceWithNodeOfFull) {
    auto topo = builder_.GetGraph()->topo.get();
    cout << utils::ToGraphviz(topo) << endl;
    auto ret = topo->ReplaceWithNode("testnode", ir::Node::Type("test", "op1", 1));
    EXPECT_EQ(RC_SUCCESS, ret);
    EXPECT_EQ(1, topo->GetInputCount());
    EXPECT_EQ(string("in1"), topo->GetEdge(topo->GetInput(0))->GetName());
    EXPECT_EQ(1, topo->GetOutputCount());
    EXPECT_EQ(string("out6"), topo->GetEdge(topo->GetOutput(0))->GetName());
    cout << utils::ToGraphviz(topo) << endl;

    uint32_t node_count = 0;
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        ++node_count;
    }
    EXPECT_EQ(1, node_count);
}

TEST_F(GraphTopoTest, ReplaceWithNodeOfPartial) {
    auto topo = builder_.GetGraph()->topo.get();

    ir::PartialGraphTopo partial_topo(topo, {topo->GetNode("b")->GetId(), topo->GetNode("d")->GetId(),
            topo->GetNode("e")->GetId()});
    cout << utils::ToGraphviz(&partial_topo) << endl;

    set<edgeid_t> in_ids = {topo->GetEdge("out1")->GetId(), topo->GetEdge("out3")->GetId()};
    EXPECT_EQ(2, partial_topo.GetInputCount());
    auto ref = in_ids.find(partial_topo.GetInput(0));
    EXPECT_NE(in_ids.end(), ref);
    in_ids.erase(ref);
    ref = in_ids.find(partial_topo.GetInput(1));
    EXPECT_NE(in_ids.end(), ref);
    in_ids.erase(ref);

    set<edgeid_t> out_ids = {topo->GetEdge("out2")->GetId(), topo->GetEdge("out4")->GetId(),
        topo->GetEdge("out5")->GetId()};
    EXPECT_EQ(3, partial_topo.GetOutputCount());
    ref = out_ids.find(partial_topo.GetOutput(0));
    EXPECT_NE(out_ids.end(), ref);
    out_ids.erase(ref);
    ref = out_ids.find(partial_topo.GetOutput(1));
    EXPECT_NE(out_ids.end(), ref);
    out_ids.erase(ref);
    ref = out_ids.find(partial_topo.GetOutput(2));
    EXPECT_NE(out_ids.end(), ref);
    out_ids.erase(ref);

    auto ret = partial_topo.ReplaceWithNode("testnode", ir::Node::Type("test", "op1", 1));
    EXPECT_EQ(RC_SUCCESS, ret);
    cout << utils::ToGraphviz(topo) << endl;
    auto testnode = topo->GetNode("testnode");
    EXPECT_NE(nullptr, testnode);

    auto prevs = topo->FindPredecessors(testnode->GetId());
    EXPECT_EQ(2, prevs.size());

    if (prevs[0] == topo->GetNode("c")->GetId()) {
        EXPECT_EQ(topo->GetNode("a")->GetId(), prevs[1]);
    } else if (prevs[0] == topo->GetNode("a")->GetId()) {
        EXPECT_EQ(topo->GetNode("c")->GetId(), prevs[1]);
    } else {
        EXPECT_TRUE(false);
    }

    auto nexts = topo->FindSuccessors(testnode->GetId());
    EXPECT_EQ(1, nexts.size());
    EXPECT_EQ(topo->GetNode("f")->GetId(), nexts[0]);
}

TEST_F(GraphTopoTest, FindAncestors) {
    auto topo = builder_.GetGraph()->topo.get();
    cout << utils::ToGraphviz(topo) << endl;
    auto ancestors = topo->FindAncestors(topo->GetNode("f")->GetId());
    EXPECT_EQ(5, ancestors.size());
    auto ref = ancestors.find(topo->GetNode("a")->GetId());
    EXPECT_NE(ancestors.end(), ref);
    ref = ancestors.find(topo->GetNode("b")->GetId());
    EXPECT_NE(ancestors.end(), ref);
    ref = ancestors.find(topo->GetNode("c")->GetId());
    EXPECT_NE(ancestors.end(), ref);
    ref = ancestors.find(topo->GetNode("d")->GetId());
    EXPECT_NE(ancestors.end(), ref);
    ref = ancestors.find(topo->GetNode("e")->GetId());
    EXPECT_NE(ancestors.end(), ref);
}

TEST_F(GraphTopoTest, TopologicalSort) {
    auto topo = builder_.GetGraph()->topo.get();
    cout << utils::ToGraphviz(topo) << endl;

    vector<nodeid_t> sorted_nodes;
    topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    auto a_idx = utils::VectorFind(sorted_nodes, topo->GetNode("a")->GetId());
    auto c_idx = utils::VectorFind(sorted_nodes, topo->GetNode("c")->GetId());
    auto e_idx = utils::VectorFind(sorted_nodes, topo->GetNode("e")->GetId());
    auto f_idx = utils::VectorFind(sorted_nodes, topo->GetNode("f")->GetId());
    EXPECT_LT(a_idx, c_idx);
    EXPECT_LT(c_idx, f_idx);
    EXPECT_LT(e_idx, f_idx);
}
