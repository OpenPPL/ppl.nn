#include "ppl/nn/models/op_info_manager.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;

static bool TestParamEqualFunc(void* param_0, void* param_1) {
    return true;
}

TEST(OpInfoManagerTest, misc) {
    OpInfoManager mgr;
    OpInfo info;
    info.param_equal = TestParamEqualFunc;
    mgr.Register("domain", "type", info);
    auto ret = mgr.Find("domain", "type");
    EXPECT_NE(nullptr, ret);
    EXPECT_EQ(TestParamEqualFunc, ret->param_equal);
}
