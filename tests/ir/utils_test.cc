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

#include "gtest/gtest.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/auxtools/to_graphviz.h"
#include "tests/ir/graph_builder.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;

class IrUtilsTest : public testing::Test {
protected:
    virtual void SetUp() override {
        builder_.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1", "out2"});
        builder_.AddNode("b", ir::Node::Type("test", "op2", 1), {"out1", "out6", "out9", "out11"}, {"out3"});
        builder_.AddNode("c", ir::Node::Type("test", "op2", 1), {"out1"}, {"out4"});
        builder_.AddNode("d", ir::Node::Type("test", "op2", 1), {"out2"}, {"out5"});
        builder_.AddNode("e", ir::Node::Type("test", "op1", 1), {"out4"}, {"out6", "out7"});
        builder_.AddNode("f", ir::Node::Type("test", "op1", 1), {"out5"}, {"out8"});
        builder_.AddNode("g", ir::Node::Type("test", "op2", 1), {"out7", "out8"}, {"out9"});
        builder_.AddNode("h", ir::Node::Type("test", "op2", 1), {"in2"}, {"out10"});
        builder_.AddNode("i", ir::Node::Type("test", "op2", 1), {"out9", "out10"}, {"out11"});
        builder_.Finalize();

        auto topo = builder_.GetGraph()->topo.get();
        cout << utils::ToGraphviz(topo) << endl;
    }

    GraphBuilder builder_;
};

TEST_F(IrUtilsTest, Dfs) {
    auto topo = builder_.GetGraph()->topo.get();
    vector<nodeid_t> sorted_nodes;

    auto node_iter = topo->CreateNodeIter();
    utils::Dfs(
        topo->GetMaxNodeId(),
        [&node_iter]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto node = node_iter->Get();
                node_iter->Forward();
                return node->GetId();
            }
            return INVALID_NODEID;
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        [topo, &sorted_nodes](nodeid_t nid) -> void {
            cout << topo->GetNodeById(nid)->GetName() << " -> ";
            sorted_nodes.push_back(nid);
        });
    cout << "nil" << endl;

    auto node_c = topo->GetNodeByName("c");
    EXPECT_TRUE(node_c != nullptr);
    auto node_g = topo->GetNodeByName("g");
    EXPECT_TRUE(node_g != nullptr);

    uint32_t node_c_pos = 0;
    for (; node_c_pos < sorted_nodes.size(); ++node_c_pos) {
        if (sorted_nodes[node_c_pos] == node_c->GetId()) {
            break;
        }
    }
    EXPECT_TRUE(node_c_pos < sorted_nodes.size());

    uint32_t node_g_pos = 0;
    for (; node_g_pos < sorted_nodes.size(); ++node_g_pos) {
        if (sorted_nodes[node_g_pos] == node_g->GetId()) {
            break;
        }
    }
    EXPECT_TRUE(node_g_pos < sorted_nodes.size());

    EXPECT_TRUE(node_c_pos < node_g_pos);

}

TEST_F(IrUtilsTest, Bfs) {
    auto topo = builder_.GetGraph()->topo.get();
    vector<nodeid_t> sorted_nodes;

    auto node_iter = topo->CreateNodeIter();
    utils::Bfs(
        topo->GetMaxNodeId(),
        [&node_iter]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto node = node_iter->Get();
                node_iter->Forward();
                return node->GetId();
            }
            return INVALID_NODEID;
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto nexts = topo->FindSuccessors(nid);
            for (auto x : nexts) {
                f(x);
            }
        },
        [topo, &sorted_nodes](nodeid_t nid, uint32_t) -> void {
            cout << topo->GetNodeById(nid)->GetName() << " -> ";
            sorted_nodes.push_back(nid);
        });
    cout << "nil" << endl;

    auto node_c = topo->GetNodeByName("c");
    EXPECT_TRUE(node_c != nullptr);
    auto node_g = topo->GetNodeByName("g");
    EXPECT_TRUE(node_g != nullptr);

    uint32_t node_c_pos = 0;
    for (; node_c_pos < sorted_nodes.size(); ++node_c_pos) {
        if (sorted_nodes[node_c_pos] == node_c->GetId()) {
            break;
        }
    }
    EXPECT_TRUE(node_c_pos < sorted_nodes.size());

    uint32_t node_g_pos = 0;
    for (; node_g_pos < sorted_nodes.size(); ++node_g_pos) {
        if (sorted_nodes[node_g_pos] == node_g->GetId()) {
            break;
        }
    }
    EXPECT_TRUE(node_g_pos < sorted_nodes.size());

    EXPECT_TRUE(node_c_pos < node_g_pos);
}

TEST_F(IrUtilsTest, DfsWithPriority) {
    auto topo = builder_.GetGraph()->topo.get();

    auto node_iter = topo->CreateNodeIter();
    vector<uint32_t> nid2level(topo->GetMaxNodeId(), 0);
    utils::Bfs(
        topo->GetMaxNodeId(),
        [&node_iter]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto node = node_iter->Get();
                node_iter->Forward();
                return node->GetId();
            }
            return INVALID_NODEID;
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto nexts = topo->FindSuccessors(nid);
            for (auto x : nexts) {
                f(x);
            }
        },
        [topo, &nid2level](nodeid_t nid, uint32_t level) -> void {
            cout << "set node[" << topo->GetNodeById(nid)->GetName() << "] level = " << level << endl;
            nid2level[nid] = level;
        });

    vector<nodeid_t> sorted_nodes;
    node_iter = topo->CreateNodeIter();
    utils::Dfs(
        topo->GetMaxNodeId(),
        [&node_iter]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto node = node_iter->Get();
                node_iter->Forward();
                return node->GetId();
            }
            return INVALID_NODEID;
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        [topo, &nid2level, &sorted_nodes](nodeid_t nid) -> void {
            cout << topo->GetNodeById(nid)->GetName() << "[" << nid2level[nid] << "] -> ";
            sorted_nodes.push_back(nid);
        },
        [&nid2level](nodeid_t a, nodeid_t b) -> bool {
            return (nid2level[a] < nid2level[b]);
        });
    cout << "nil" << endl;

    auto node_a = topo->GetNodeByName("a");
    EXPECT_TRUE(node_a != nullptr);
    auto node_h = topo->GetNodeByName("h");
    EXPECT_TRUE(node_h != nullptr);

    uint32_t node_a_pos = 0;
    for (; node_a_pos < sorted_nodes.size(); ++node_a_pos) {
        if (sorted_nodes[node_a_pos] == node_a->GetId()) {
            break;
        }
    }
    EXPECT_TRUE(node_a_pos < sorted_nodes.size());

    uint32_t node_h_pos = 0;
    for (; node_h_pos < sorted_nodes.size(); ++node_h_pos) {
        if (sorted_nodes[node_h_pos] == node_h->GetId()) {
            break;
        }
    }
    EXPECT_TRUE(node_h_pos < sorted_nodes.size());

    EXPECT_TRUE(node_a_pos < node_h_pos);
}
