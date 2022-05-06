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
#include "tests/ir/graph_builder.h"
#include "tests/engines/tmp_engine.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/auxtools/to_graphviz.h"
#include "ppl/nn/optimizers/engine_graph_partitioner.h"
#include "ppl/nn/common/logger.h"
#include <utility>
#include <memory>
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class OptimizerUtilsTest : public testing::Test {
protected:
    void SetUp() {
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngine1()));
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngine2()));

        resource_.engines.resize(2);
        resource_.engines[0] = engines_[0].get();
        resource_.engines[1] = engines_[1].get();
        resource_.graph_partitioner = make_shared<EngineGraphPartitioner>();
    }

    vector<unique_ptr<EngineImpl>> engines_;
    utils::SharedResource resource_;
};

TEST_F(OptimizerUtilsTest, basic_partition) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op2", 1), {"out1"}, {"out2"});
    builder.AddNode("c", ir::Node::Type("test", "op1", 1), {"out2"}, {"out3"});
    builder.AddNode("d", ir::Node::Type("test", "op2", 1), {"out3"}, {"out4"});
    builder.Finalize();

    auto graph = builder.GetGraph();
    auto topo = graph->topo.get();

    auto graph_info = make_shared<RuntimeGraphInfo>();
    auto status = utils::ProcessGraph(resource_, graph, graph_info.get());
    EXPECT_EQ(RC_SUCCESS, status);

    LOG(DEBUG) << utils::ToGraphviz(topo);

    auto node_b = topo->GetNode("b");
    EXPECT_TRUE(node_b != nullptr);

    auto next_ids = topo->FindSuccessors(node_b->GetId());
    EXPECT_EQ(1, next_ids.size());

    auto next_of_b = topo->GetNode(next_ids[0]);
    EXPECT_TRUE(next_of_b != nullptr);

    auto& type = next_of_b->GetType();
    EXPECT_EQ("pmx", type.domain);
    EXPECT_EQ("Converter", type.name);
    EXPECT_EQ(1, type.version);
}

TEST_F(OptimizerUtilsTest, converters_for_input) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op2", 1), {"in1", "out1"}, {"out2"});
    builder.Finalize();

    auto graph = builder.GetGraph();
    auto topo = graph->topo.get();

    auto graph_info = make_shared<RuntimeGraphInfo>();
    auto status = utils::ProcessGraph(resource_, graph, graph_info.get());
    EXPECT_EQ(RC_SUCCESS, status);

    LOG(DEBUG) << utils::ToGraphviz(topo);

    auto edge_in1 = topo->GetEdge("in1");
    EXPECT_TRUE(edge_in1 != nullptr);
    uint32_t consumer_count = edge_in1->CalcConsumerCount();
    EXPECT_EQ(2, consumer_count);

    uint32_t converter_count = 0;
    for (auto it = edge_in1->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto nid = it.Get();
        auto node = topo->GetNode(nid);
        auto& type = node->GetType();
        if (type.domain == "pmx" && type.name == "Converter" && type.version == 1) {
            ++converter_count;
        }
    }
    EXPECT_TRUE(converter_count == 1);
}

#if 0
TEST_F(OptimizerUtilsTest, partition_sorting) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1", "out2"});
    builder.AddNode("b", ir::Node::Type("test", "op2", 1), {"out1", "out6", "out9", "out11"}, {"out3"});
    builder.AddNode("c", ir::Node::Type("test", "op1", 1), {"out1"}, {"out4"});
    builder.AddNode("d", ir::Node::Type("test", "op1", 1), {"out2"}, {"out5"});
    builder.AddNode("e", ir::Node::Type("test", "op1", 1), {"out4"}, {"out6", "out7"});
    builder.AddNode("f", ir::Node::Type("test", "op1", 1), {"out5"}, {"out8"});
    builder.AddNode("g", ir::Node::Type("test", "op1", 1), {"out7", "out8"}, {"out9"});
    builder.AddNode("h", ir::Node::Type("test", "op2", 1), {"in2"}, {"out10"});
    builder.AddNode("i", ir::Node::Type("test", "op2", 1), {"out9", "out10"}, {"out11"});
    builder.Finalize();

    auto graph = builder.GetGraph();
    auto topo = graph->topo.get();

    auto node_a = topo->GetNode("a");
    EXPECT_TRUE(node_a != nullptr);
    auto node_b = topo->GetNode("b");
    EXPECT_TRUE(node_b != nullptr);
    auto node_c = topo->GetNode("c");
    EXPECT_TRUE(node_c != nullptr);
    auto node_d = topo->GetNode("d");
    EXPECT_TRUE(node_d != nullptr);
    auto node_e = topo->GetNode("e");
    EXPECT_TRUE(node_e != nullptr);
    auto node_f = topo->GetNode("f");
    EXPECT_TRUE(node_f != nullptr);
    auto node_g = topo->GetNode("g");
    EXPECT_TRUE(node_g != nullptr);
    auto node_h = topo->GetNode("h");
    EXPECT_TRUE(node_h != nullptr);
    auto node_i = topo->GetNode("i");
    EXPECT_TRUE(node_i != nullptr);

    RuntimeGraphInfo graph_info;
    auto status = utils::ProcessGraph(resource_, graph, &graph_info);
    EXPECT_EQ(RC_SUCCESS, status);

    LOG(DEBUG) << utils::ToGraphviz(topo);

    LOG(DEBUG) << "number of partitions = " << graph_info.partitions.size();
    for (uint32_t i = 0; i < graph_info.partitions.size(); ++i) {
        auto& partition = graph_info.partitions[i];
        LOG(DEBUG) << "partition [" << i << "], number of nodes [" << partition.ops.size() << "]:";
        vector<nodeid_t> sorted_nodes(partition.ops.size());
        for (uint32_t j = 0; j < partition.ops.size(); ++j) {
            auto& op = partition.ops[j];
            sorted_nodes[j] = op->GetNode()->GetId();
            LOG(DEBUG) << "    " << op->GetNode()->GetName();
        }

        if (i == 0) {
            EXPECT_TRUE(utils::VectorFind(sorted_nodes, node_a->GetId()) < sorted_nodes.size());
        } else if (i == graph_info.partitions.size() - 1) {
            EXPECT_TRUE(utils::VectorFind(sorted_nodes, node_b->GetId()) < sorted_nodes.size());
        }
    }
}
#endif
