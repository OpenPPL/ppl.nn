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
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;

class PartialGraphTopoTest : public testing::Test {};

TEST_F(PartialGraphTopoTest, constructor) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"});
    builder.AddNode("c", ir::Node::Type("test", "op1", 1), {"out2"}, {"out3"});
    builder.AddNode("d", ir::Node::Type("test", "op1", 1), {"out3"}, {"out4"});
    builder.AddNode("e", ir::Node::Type("test", "op1", 1), {"out4"}, {"out5"});
    builder.Finalize();

    auto parent_topo = builder.GetGraph()->topo.get();

    set<string> nodes = {"b", "c", "d"};
    auto nid_b = parent_topo->GetNode("b")->GetId();
    auto nid_c = parent_topo->GetNode("c")->GetId();
    auto nid_d = parent_topo->GetNode("d")->GetId();

    ir::PartialGraphTopo partial_topo(parent_topo, {nid_b, nid_c, nid_d});

    EXPECT_EQ(1, partial_topo.GetInputCount());
    auto edge = partial_topo.GetEdge(partial_topo.GetInput(0));
    EXPECT_EQ(string("out1"), edge->GetName());

    EXPECT_EQ(1, partial_topo.GetOutputCount());
    edge = partial_topo.GetEdge(partial_topo.GetOutput(0));
    EXPECT_EQ(string("out4"), edge->GetName());

    uint32_t node_counter = 0;
    for (auto it = partial_topo.CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        EXPECT_TRUE(nodes.find(node->GetName()) != nodes.end());
        ++node_counter;
    }
    EXPECT_TRUE(node_counter == nodes.size());
}

TEST_F(PartialGraphTopoTest, extra_input_1) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"}, {"extra_b1", "in1"});
    builder.AddNode("c", ir::Node::Type("test", "op1", 1), {"out2"}, {"out3"});
    builder.Finalize();

    auto topo = builder.GetGraph()->topo.get();
    EXPECT_EQ(1, topo->GetExtraInputCount());

    auto nodeid_a = topo->GetNode("a")->GetId();
    auto nodeid_b = topo->GetNode("b")->GetId();
    ir::PartialGraphTopo partial_topo(topo, {nodeid_a, nodeid_b});

    EXPECT_EQ(1, partial_topo.GetInputCount());
    EXPECT_EQ(string("in1"), partial_topo.GetEdge(partial_topo.GetInput(0))->GetName());

    EXPECT_EQ(1, partial_topo.GetOutputCount());
    EXPECT_EQ(string("out2"), partial_topo.GetEdge(partial_topo.GetOutput(0))->GetName());

    set<string> expected_extra_inputs = {"extra_b1"};
    EXPECT_EQ(1, partial_topo.GetExtraInputCount());
    EXPECT_EQ(string("extra_b1"), partial_topo.GetEdge(partial_topo.GetExtraInput(0))->GetName());
}

TEST_F(PartialGraphTopoTest, extra_input_2) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"}, {"extra_b1", "extra_b2"});
    builder.AddNode("c", ir::Node::Type("test", "op1", 1), {"out2"}, {"out3"});
    builder.Finalize();

    auto topo = builder.GetGraph()->topo.get();
    set<string> expected_extra_inputs = {"extra_b1", "extra_b2"};

    EXPECT_EQ(2, topo->GetExtraInputCount());

    auto nodeid_b = topo->GetNode("b")->GetId();
    ir::PartialGraphTopo partial_topo(topo, {nodeid_b});

    EXPECT_EQ(1, partial_topo.GetInputCount());
    EXPECT_EQ(string("out1"), partial_topo.GetEdge(partial_topo.GetInput(0))->GetName());

    EXPECT_EQ(1, partial_topo.GetOutputCount());
    EXPECT_EQ(string("out2"), partial_topo.GetEdge(partial_topo.GetOutput(0))->GetName());

    EXPECT_EQ(2, partial_topo.GetExtraInputCount());
    for (uint32_t i = 0; i < partial_topo.GetExtraInputCount(); ++i) {
        auto eid = partial_topo.GetExtraInput(i);
        auto edge = partial_topo.GetEdge(eid);
        EXPECT_TRUE(expected_extra_inputs.find(edge->GetName()) != expected_extra_inputs.end());
    }
}
