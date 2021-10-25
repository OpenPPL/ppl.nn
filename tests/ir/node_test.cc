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

#include "ppl/nn/ir/node.h"
#include "gtest/gtest.h"

using namespace std;
using namespace ppl::nn;

TEST(NodeTest, NodeTest_GetId_Test) {
    const nodeid_t nodeid = 1;
    ir::Node node(nodeid);
    EXPECT_EQ(node.GetId(), nodeid);
}

TEST(NodeTest, NodeTest_SetNameAndGetName_Test) {
    const string node_name = "tmp";
    ir::Node node(1);
    node.SetName(node_name);
    EXPECT_EQ(node_name, node.GetName());
}

TEST(NodeTest, NodeTest_SetTypeAndGetType_Test) {
    ir::Node node(1);
    node.SetType(ir::Node::Type("domain", "test", 1));
    const ir::Node::Type& type = node.GetType();
    EXPECT_EQ("domain", type.domain);
    EXPECT_EQ("test", type.name);
}

TEST(NodeTest, NodeTest_AddInputAndGetInput_Test) {
    ir::Node node(1);
    const edgeid_t expected_edgeid = 2;
    node.AddInput(expected_edgeid);
    EXPECT_EQ(expected_edgeid, node.GetInput(0));
}

TEST(NodeTest, NodeTest_GetInputCount_Test) {
    ir::Node node(1);
    const edgeid_t expected_edgeid = 2;
    node.AddInput(expected_edgeid);
    EXPECT_EQ(1, node.GetInputCount());
}

TEST(NodeTest, NodeTest_ReplaceInput_Test) {
    ir::Node node(1);
    node.AddInput(2);
    EXPECT_EQ(1, node.ReplaceInput(2, 4));
    EXPECT_EQ(4, node.GetInput(0));
}

TEST(NodeTest, NodeTest_AddOutputAndGetOutput_Test) {
    ir::Node node(1);
    node.AddOutput(2);
    EXPECT_EQ(2, node.GetOutput(0));
}

TEST(NodeTest, NodeTest_GerOutputCount_Test) {
    ir::Node node(1);
    node.AddOutput(2);
    EXPECT_EQ(1, node.GetOutputCount());
}

TEST(NodeTest, NodeTest_ReplaceOutput__Test) {
    ir::Node node(1);
    node.AddOutput(2);
    EXPECT_EQ(1, node.ReplaceOutput(2, 4));
    EXPECT_EQ(4, node.GetOutput(0));
}

TEST(NodeTest, NodeTest_AddExtralInputAndGetExtraInput_Test) {
    ir::Node node(1);
    node.AddExtraInput(2);
    EXPECT_EQ(2, node.GetExtraInput(0));
}

TEST(NodeTest, NodeTest_GetExtraInputCount_Test) {
    ir::Node node(1);
    node.AddExtraInput(2);
    EXPECT_EQ(1, node.GetExtraInputCount());
}

TEST(NodeTest, NodeTest_ReplaceExtraInput__Test) {
    ir::Node node(1);
    node.AddExtraInput(2);
    EXPECT_EQ(1, node.ReplaceExtraInput(2, 4));
    EXPECT_EQ(4, node.GetExtraInput(0));
}
