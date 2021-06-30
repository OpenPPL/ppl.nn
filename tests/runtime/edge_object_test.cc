#include "ppl/nn/runtime/edge_object.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;

class EdgeObjectTest : public testing::Test {
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

TEST_F(EdgeObjectTest, misc) {
    auto topo = builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(0);
    EXPECT_NE(nullptr, edge);

    EdgeObject object(edge, EdgeObject::T_EDGE_OBJECT);
    EXPECT_EQ(edge, object.GetEdge());
    EXPECT_EQ(EdgeObject::T_EDGE_OBJECT, object.GetObjectType());
}
