#include "ppl/nn/runtime/kernel_exec_context.h"
#include "tests/ir/graph_builder.h"
#include "tests/runtime/test_barrier.h"
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

        barriers_.resize(builder_.GetGraph()->topo->GetMaxEdgeId());
    }

protected:
    GraphBuilder builder_;
    vector<TestBarrier> barriers_;
};

TEST_F(KernelExecContextTest, misc) {
    auto topo = builder_.GetGraph()->topo.get();

    auto node = topo->GetNodeById(0);
    EXPECT_EQ("a", node->GetName());

    KernelExecContext ctx;
    ctx.SetNode(node);
    ctx.SetGetBarrierFunc([this](edgeid_t eid) -> Barrier* {
        return &barriers_[eid];
    });

    auto edge = topo->GetEdgeByName("input_of_a");
    EXPECT_NE(nullptr, edge);
    EXPECT_EQ(&barriers_[0], ctx.GetInputBarrier(0));

    edge = topo->GetEdgeByName("output_of_a");
    EXPECT_NE(nullptr, edge);
    EXPECT_EQ(&barriers_[1], ctx.GetOutputBarrier(0));
}
