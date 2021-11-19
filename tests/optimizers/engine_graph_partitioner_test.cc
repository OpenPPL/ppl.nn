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

#include "gtest/gtest.h"
#include "tests/ir/graph_builder.h"
#include "tests/engines/tmp_engine.h"

#include <vector>
#include <memory>

using namespace std;
using namespace ppl::nn;
using namespace ppl::common;
using namespace ppl::nn::test;

class TestGraphPartion : public testing::Test {
protected:
    virtual void SetUp() override {
        builder_.AddNode("a", ir::Node::Type("test", "op1", 1), {"input_of_a"}, {"output_of_a"});
        builder_.AddNode("b", ir::Node::Type("test", "op1", 1), {"output_of_a"}, {"output_of_b"});
        builder_.AddNode("c", ir::Node::Type("test", "op2", 1), {"output_of_b"}, {"output_of_c"});
        builder_.AddNode("d", ir::Node::Type("test", "op2", 1), {"output_of_c"}, {"output_of_d"});
        builder_.Finalize();
    }

    GraphBuilder builder_;
};

TEST_F(TestGraphPartion, partition1) {
    vector<unique_ptr<EngineImpl>> engines;
    engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngineOne()));
    engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngineTwo()));

    auto resource = make_shared<utils::SharedResource>();
    resource->engines.resize(2);
    resource->engines[0] = engines[0].get();
    resource->engines[1] = engines[1].get();

    EngineGraphPartitioner partitioner;
    vector<pair<EngineImpl*, vector<nodeid_t>>> partitions;
    auto status = partitioner.Partition(resource.get(), builder_.GetGraph()->topo.get(), &partitions);
    EXPECT_EQ(status, RC_SUCCESS);
    EXPECT_EQ(partitions.size(), 2);

    for (auto partition : partitions) {
        if (string(partition.first->GetName()) == "tmpOne") {
            for (auto node_id : partition.second) {
                EXPECT_TRUE(node_id == 0 || node_id == 1);
            }
        }
        if (string(partition.first->GetName()) == "tmpTwo") {
            for (auto node_id : partition.second) {
                EXPECT_TRUE(node_id == 2 || node_id == 3);
            }
        }
    }
}
