#include "ppl/nn/common/input_output_info.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
#include <vector>
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class InputOutputInfoTest : public testing::Test {
protected:
    void SetUp() override {
        builder_.AddNode("a", ir::Node::Type("test", "op1"), {"input_of_a"}, {"output_of_a"});
        builder_.AddNode("b", ir::Node::Type("test", "op2"), {"output_of_a"}, {"output_of_b"});
        builder_.AddNode("c", ir::Node::Type("test", "op3"), {"output_of_b"}, {"output_of_c"});
        builder_.Finalize();

        auto topo = builder_.GetGraph()->topo.get();
        edge_objects_.reserve(topo->GetMaxEdgeId());
        for (edgeid_t i = 0; i < topo->GetMaxEdgeId(); ++i) {
            auto edge = topo->GetEdgeById(i);
            if (!edge) {
                continue;
            }
            edge_objects_.emplace_back(EdgeObject(edge, EdgeObject::T_EDGE_OBJECT));
        }
    }

protected:
    GraphBuilder builder_;
    vector<EdgeObject> edge_objects_;
};

TEST_F(InputOutputInfoTest, misc) {
    auto topo = builder_.GetGraph()->topo.get();

    auto node = topo->GetNodeById(0);
    EXPECT_EQ("a", node->GetName());

    InputOutputInfo info;
    info.SetNode(node);
    info.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
        return &edge_objects_[eid];
    });

    auto edge = topo->GetEdgeByName("input_of_a");
    EXPECT_NE(nullptr, edge);
    EXPECT_EQ(&edge_objects_[0], info.GetInput<EdgeObject>(0));

    edge = topo->GetEdgeByName("output_of_a");
    EXPECT_NE(nullptr, edge);
    EXPECT_EQ(&edge_objects_[1], info.GetOutput<EdgeObject>(0));
}
