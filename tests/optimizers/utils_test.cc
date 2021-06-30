#include "ppl/nn/optimizers/utils.h"
#include "gtest/gtest.h"
#include "tests/ir/graph_builder.h"
#include "tests/engines/tmp_engine.h"
#include <iostream>
#include <utility>
#include <memory>
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class UtilsTest : public testing::Test {
protected:
    virtual void SetUp() override {
        builder_.SetGraphName("tmp");
        builder_.AddNode("a", ir::Node::Type("test", "op1"), {"input_of_a"}, {"output_of_a"});
        builder_.AddNode("b", ir::Node::Type("test", "op2"), {"output_of_a"}, {"output_of_b"});
        builder_.AddNode("c", ir::Node::Type("test", "op3"), {"output_of_b"}, {"output_of_c"});
        builder_.AddNode("d", ir::Node::Type("test", "op4"), {"output_of_c"}, {"output_of_d"});
        builder_.Finalize();
    }
    GraphBuilder builder_;
};

TEST_F(UtilsTest, basic_partition) {
    auto resource = make_shared<utils::SharedResource>();
    auto graph_info = make_shared<RuntimeGraphInfo>();
    resource->engines.reserve(2);
    resource->engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngineOne()));
    resource->engines.emplace_back(unique_ptr<EngineImpl>(new TmpEngineTwo()));
    auto status = utils::ProcessGraph(resource.get(), builder_.GetGraph(), graph_info.get());
    EXPECT_EQ(status, RC_SUCCESS);
}
