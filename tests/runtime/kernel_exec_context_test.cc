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

#include "ppl/nn/runtime/kernel_exec_context.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
#include <vector>
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class KernelExecContextTest : public testing::Test {
protected:
    void SetUp() override {
        builder_.AddNode("a", ir::Node::Type("test", "op1"), {"input_of_a"}, {"output_of_a"});
        builder_.AddNode("b", ir::Node::Type("test", "op2"), {"output_of_a"}, {"output_of_b"});
        builder_.AddNode("c", ir::Node::Type("test", "op3"), {"output_of_b"}, {"output_of_c"});
        builder_.Finalize();
    }

protected:
    GraphBuilder builder_;
};

TEST_F(KernelExecContextTest, misc) {
    auto topo = builder_.GetGraph()->topo.get();

    auto node = topo->GetNodeById(0);
    EXPECT_EQ("a", node->GetName());

    KernelExecContext ctx;
    ctx.SetNode(node);

    auto edge = topo->GetEdgeByName("input_of_a");
    EXPECT_NE(nullptr, edge);

    edge = topo->GetEdgeByName("output_of_a");
    EXPECT_NE(nullptr, edge);
}
