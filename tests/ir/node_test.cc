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
    node.SetType(ir::Node::Type("domain", "test"));
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
