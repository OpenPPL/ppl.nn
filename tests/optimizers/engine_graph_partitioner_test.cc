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

#include "ppl/nn/optimizers/engine_graph_partitioner.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/auxtools/to_graphviz.h"
#include "gtest/gtest.h"
#include "tests/ir/graph_builder.h"
#include "tests/engines/tmp_engine.h"
#include <vector>
#include <memory>

using namespace std;
using namespace ppl::nn;
using namespace ppl::common;
using namespace ppl::nn::test;

class TestEngineGraphPartioner : public testing::Test {
protected:
    void SetUp() override {
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngine1()));
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngine2()));
        engines_.emplace_back(unique_ptr<EngineImpl>(new TmpEngine3()));

        engine_ptrs_.resize(engines_.size());
        for (uint32_t i = 0; i < engines_.size(); ++i) {
            engine_ptrs_[i] = engines_[i].get();
        }
    }

    vector<unique_ptr<EngineImpl>> engines_;
    vector<EngineImpl*> engine_ptrs_;
};

TEST_F(TestEngineGraphPartioner, partitioned_by_engine) {
    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"});
    builder.AddNode("c", ir::Node::Type("test", "op2", 1), {"out1"}, {"out3"});
    builder.AddNode("d", ir::Node::Type("test", "op2", 1), {"out3"}, {"out4"});
    builder.AddNode("e", ir::Node::Type("test", "op3", 1), {"out1"}, {"out5"});
    builder.AddNode("f", ir::Node::Type("test", "op3", 1), {"out2", "out4", "out5"}, {"out6"});
    builder.Finalize();
    auto topo = builder.GetGraph()->topo.get();
    cout << utils::ToGraphviz(topo) << endl;

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

    EngineGraphPartitioner partitioner;
    vector<pair<EngineImpl*, vector<nodeid_t>>> partitions;
    auto status = partitioner.Partition(engine_ptrs_, builder.GetGraph()->topo.get(), &partitions);
    EXPECT_EQ(RC_SUCCESS, status);
    EXPECT_EQ(3, partitions.size());

    for (uint32_t i = 0; i < partitions.size(); ++i) {
        auto& node_ids = partitions[i].second;

        cout << "partition[" << i << "] with engine[" << partitions[i].first->GetName() << "]: ";
        for (auto x = partitions[i].second.begin(); x != partitions[i].second.end(); ++x) {
            cout << topo->GetNode(*x)->GetName() << " ";
        }
        cout << endl;

        EXPECT_EQ(2, node_ids.size());
        if (utils::VectorFind(node_ids, node_a->GetId()) < node_ids.size()) {
            EXPECT_TRUE(utils::VectorFind(node_ids, node_b->GetId()) < node_ids.size());
        } else if (utils::VectorFind(node_ids, node_c->GetId()) < node_ids.size()) {
            EXPECT_TRUE(utils::VectorFind(node_ids, node_d->GetId()) < node_ids.size());
        } else if (utils::VectorFind(node_ids, node_e->GetId()) < node_ids.size()) {
            EXPECT_TRUE(utils::VectorFind(node_ids, node_f->GetId()) < node_ids.size());
        }
    }
}

TEST_F(TestEngineGraphPartioner, partition_root_node) {
    vector<unique_ptr<EngineImpl>> engines;
    engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngine1()));
    engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngine2()));

    vector<EngineImpl*> engine_ptrs(engines.size());
    for (uint32_t i = 0; i < engines.size(); ++i) {
        engine_ptrs[i] = engines[i].get();
    }

    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"});
    builder.AddNode("c", ir::Node::Type("test", "op1", 1), {"out2"}, {"out3"});
    builder.AddNode("d", ir::Node::Type("test", "op2", 1), {"in2"}, {"out4"});
    builder.AddNode("e", ir::Node::Type("test", "op2", 1), {"out4"}, {"out5"});
    builder.AddNode("f", ir::Node::Type("test", "op2", 1), {"out5"}, {"out6"});
    builder.Finalize();
    auto topo = builder.GetGraph()->topo.get();
    cout << utils::ToGraphviz(topo) << endl;

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

    EngineGraphPartitioner partitioner;
    vector<pair<EngineImpl*, vector<nodeid_t>>> partitions;
    auto status = partitioner.Partition(engine_ptrs, builder.GetGraph()->topo.get(), &partitions);
    EXPECT_EQ(RC_SUCCESS, status);
    EXPECT_EQ(2, partitions.size());

    for (uint32_t i = 0; i < partitions.size(); ++i) {
        auto& node_ids = partitions[i].second;

        cout << "partition[" << i << "] with engine[" << partitions[i].first->GetName() << "]: ";
        for (auto x = partitions[i].second.begin(); x != partitions[i].second.end(); ++x) {
            cout << topo->GetNode(*x)->GetName() << " ";
        }
        cout << endl;

        EXPECT_EQ(3, node_ids.size());
        if (utils::VectorFind(node_ids, node_a->GetId()) < node_ids.size()) {
            EXPECT_TRUE(utils::VectorFind(node_ids, node_b->GetId()) < node_ids.size());
            EXPECT_TRUE(utils::VectorFind(node_ids, node_c->GetId()) < node_ids.size());
        } else if (utils::VectorFind(node_ids, node_d->GetId()) < node_ids.size()) {
            EXPECT_TRUE(utils::VectorFind(node_ids, node_e->GetId()) < node_ids.size());
            EXPECT_TRUE(utils::VectorFind(node_ids, node_f->GetId()) < node_ids.size());
        }
    }
}

TEST_F(TestEngineGraphPartioner, partition_by_types) {
    vector<unique_ptr<EngineImpl>> engines;
    engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngine()));

    vector<EngineImpl*> engine_ptrs(engines.size());
    for (uint32_t i = 0; i < engines.size(); ++i) {
        engine_ptrs[i] = engines[i].get();
    }

    GraphBuilder builder;
    builder.AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1"});
    builder.AddNode("b", ir::Node::Type("test", "op1", 1), {"out1"}, {"out2"});
    builder.AddNode("c", ir::Node::Type("test", "op2", 1), {"out1"}, {"out3"});
    builder.AddNode("d", ir::Node::Type("test", "op2", 1), {"out3"}, {"out4"});
    builder.AddNode("e", ir::Node::Type("test", "op3", 1), {"out1"}, {"out5"});
    builder.AddNode("f", ir::Node::Type("test", "op3", 1), {"out2", "out4", "out5"}, {"out6"});
    builder.Finalize();
    auto topo = builder.GetGraph()->topo.get();
    cout << utils::ToGraphviz(topo) << endl;

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

    EngineGraphPartitioner partitioner;
    partitioner.SetSpecialTypes({{"test", "op2"}});

    vector<pair<EngineImpl*, vector<nodeid_t>>> partitions;
    auto status = partitioner.Partition(engine_ptrs, builder.GetGraph()->topo.get(), &partitions);
    EXPECT_EQ(RC_SUCCESS, status);
    EXPECT_EQ(3, partitions.size());

    for (uint32_t i = 0; i < partitions.size(); ++i) {
        auto& node_ids = partitions[i].second;

        cout << "partition[" << i << "] with engine[" << partitions[i].first->GetName() << "]: ";
        for (auto x = node_ids.begin(); x != node_ids.end(); ++x) {
            cout << topo->GetNode(*x)->GetName() << " ";
        }
        cout << endl;

        if (utils::VectorFind(node_ids, node_a->GetId()) < node_ids.size()) {
            EXPECT_EQ(3, node_ids.size());
            EXPECT_TRUE(utils::VectorFind(node_ids, node_e->GetId()) < node_ids.size());
            EXPECT_TRUE(utils::VectorFind(node_ids, node_b->GetId()) < node_ids.size());
        } else if (utils::VectorFind(node_ids, node_c->GetId()) < node_ids.size()) {
            EXPECT_EQ(2, node_ids.size());
            EXPECT_TRUE(utils::VectorFind(node_ids, node_d->GetId()) < node_ids.size());
        } else if (utils::VectorFind(node_ids, node_f->GetId()) < node_ids.size()) {
            EXPECT_EQ(1, node_ids.size());
        }
    }
}
