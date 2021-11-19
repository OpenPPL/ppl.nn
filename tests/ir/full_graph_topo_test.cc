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

#include "ppl/nn/ir/full_graph_topo.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;

class FullGraphTopoTest : public testing::Test {
protected:
    void SetUp() override {
        graph_builder_.AddNode("c", ir::Node::Type("test", "echo", 1), {"output_of_b"}, {"cc"});
        graph_builder_.AddNode("b", ir::Node::Type("test", "echo", 1), {"ab"}, {"output_of_b"});
        graph_builder_.AddNode("a", ir::Node::Type("test", "echo", 1), {"input_of_a"}, {"ab"});
        graph_builder_.Finalize();
    }

    GraphBuilder graph_builder_;
};

// test FullGraphtopo's api
TEST_F(FullGraphTopoTest, full_graph_topo_AddNode_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    const string node_name = "test_node_name";
    auto ret_pair = topo->AddNode(node_name);
    EXPECT_TRUE(ret_pair.second);
    EXPECT_EQ(ret_pair.first->GetName(), node_name);
}

TEST_F(FullGraphTopoTest, full_graph_topo_CreateNodeIter_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto iter = topo->CreateNodeIter();
    EXPECT_TRUE(iter->IsValid());
}

TEST_F(FullGraphTopoTest, full_graph_topo_GetMaxNodeId_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto res = topo->GetMaxNodeId();
    EXPECT_EQ(res, 3);
}

TEST_F(FullGraphTopoTest, full_grapg_topo_GetNodeById_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto res_node = topo->GetNodeById(0);
    cout << res_node->GetName() << endl;
    EXPECT_EQ(res_node->GetName(), "c");
}

TEST_F(FullGraphTopoTest, full_graph_topo_DelNodeById_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto ret_pair = topo->AddNode("tmp");
    EXPECT_TRUE(ret_pair.second);
    nodeid_t expected_nodeid = 3;
    EXPECT_EQ(expected_nodeid, ret_pair.first->GetId());
    topo->DelNodeById(expected_nodeid);
    auto res = topo->GetNodeById(expected_nodeid);
    EXPECT_EQ(res, nullptr);
}

TEST_F(FullGraphTopoTest, full_graph_topo_AddEdge_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    string edge_name = "tmp";
    auto ret_pair = topo->AddEdge(edge_name);
    EXPECT_TRUE(ret_pair.second);
    EXPECT_EQ(ret_pair.first->GetName(), edge_name);
}

TEST_F(FullGraphTopoTest, full_graph_topo_CreateEdgeIter_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto iter = topo->CreateEdgeIter();
    EXPECT_TRUE(iter->IsValid());
}

TEST_F(FullGraphTopoTest, full_graph_topo_GetMaxEdgeId_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto id = topo->GetMaxEdgeId();
    EXPECT_EQ(id, 4);
}

TEST_F(FullGraphTopoTest, full_graph_topo_GetEdgeById_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto res_edge = topo->GetEdgeById(0);
    EXPECT_EQ(res_edge->GetName(), "output_of_b");
}

TEST_F(FullGraphTopoTest, full_graph_topo_DelEdgeById_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto ret_pair = topo->AddEdge("tmp");
    EXPECT_TRUE(ret_pair.second);
    topo->DelEdgeById(4);
    auto res = topo->GetEdgeById(4);
    EXPECT_EQ(res, nullptr);
}

// test FullGrahpEdge's api
TEST_F(FullGraphTopoTest, full_edge_GetId_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    auto edge_id = edge->GetId();
    EXPECT_EQ(0, edge_id);
}

TEST_F(FullGraphTopoTest, full_edge_SetNameAndGetName_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    const string edge_name = "tmp";
    edge->SetName(edge_name);
    EXPECT_EQ(edge->GetName(), edge_name);
}

TEST_F(FullGraphTopoTest, full_edge_SetProducerAndGetProducer_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    edge->SetProducer(10);
    auto producer = edge->GetProducer();
    EXPECT_EQ(producer, 10);
}

TEST_F(FullGraphTopoTest, full_edge_CreateConsumerIter_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    auto consumerIter = edge->CreateConsumerIter();
    EXPECT_TRUE(consumerIter.IsValid());
}

TEST_F(FullGraphTopoTest, full_edge_CalcConsumerCount_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    auto consumers_count = edge->CalcConsumerCount();
    EXPECT_EQ(consumers_count, 1);
}

TEST_F(FullGraphTopoTest, full_edge_AddConsumer_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    edge->AddConsumer(10);
    auto constumers_count = edge->CalcConsumerCount();
    EXPECT_EQ(constumers_count, 2);
}

TEST_F(FullGraphTopoTest, full_edge_DelConsumer_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    auto consumerIter = edge->CreateConsumerIter();
    edge->DelConsumer(consumerIter.Get());
    auto constumers_count = edge->CalcConsumerCount();
    EXPECT_EQ(constumers_count, 0);
}

TEST_F(FullGraphTopoTest, full_edge_ClearConsumer_Test) {
    auto topo = graph_builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    auto consumerIter = edge->CreateConsumerIter();
    edge->ClearConsumer();
    auto constumers_count = edge->CalcConsumerCount();
    EXPECT_EQ(constumers_count, 0);
}
