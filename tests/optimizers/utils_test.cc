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

#include "ppl/nn/optimizers/utils.h"
#include "gtest/gtest.h"
#include "tests/ir/graph_builder.h"
#include "tests/engines/tmp_engine.h"
#include "ppl/nn/auxtools/to_graphviz.h"
#include "ppl/nn/common/logger.h"
#include <iostream>
#include <utility>
#include <memory>
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class OptimizerUtilsTest : public testing::Test {
protected:
    void SetUp() {
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngineOne()));
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngineTwo()));

        resource_ = make_shared<utils::SharedResource>();
        resource_->engines.resize(2);
        resource_->engines[0] = engines_[0].get();
        resource_->engines[1] = engines_[1].get();
    }

    vector<unique_ptr<EngineImpl>> engines_;
    shared_ptr<utils::SharedResource> resource_;
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
    auto status = utils::ProcessGraph(resource_.get(), graph, graph_info.get());
    EXPECT_EQ(RC_SUCCESS, status);

    LOG(DEBUG) << utils::ToGraphviz(topo);

    auto node_b = topo->GetNodeByName("b");
    EXPECT_TRUE(node_b != nullptr);

    auto next_ids = topo->FindSuccessors(node_b->GetId());
    EXPECT_EQ(1, next_ids.size());

    auto next_of_b = topo->GetNodeById(next_ids[0]);
    EXPECT_TRUE(next_of_b != nullptr);

    auto& type = next_of_b->GetType();
    EXPECT_EQ("ppl", type.domain);
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
    auto status = utils::ProcessGraph(resource_.get(), graph, graph_info.get());
    EXPECT_EQ(RC_SUCCESS, status);

    LOG(DEBUG) << utils::ToGraphviz(topo);

    auto edge_in1 = topo->GetEdgeByName("in1");
    EXPECT_TRUE(edge_in1 != nullptr);
    uint32_t consumer_count = edge_in1->CalcConsumerCount();
    EXPECT_EQ(2, consumer_count);

    uint32_t converter_count = 0;
    for (auto it = edge_in1->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto nid = it.Get();
        auto node = topo->GetNodeById(nid);
        auto& type = node->GetType();
        if (type.domain == "ppl" && type.name == "Converter" && type.version == 1) {
            ++converter_count;
        }
    }
    EXPECT_TRUE(converter_count == 1);
}
