#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/split.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_split) {
    DEFINE_ARG(SplitParam, split);
    split_param1.axis = 32;
    MAKE_BUFFER(SplitParam, split);
    int32_t axis = split_param3.axis;
    EXPECT_EQ(32, axis);
}