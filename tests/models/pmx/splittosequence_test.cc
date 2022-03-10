#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/splittosequence.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_splittosequence) {
    DEFINE_ARG(SplitToSequenceParam, splittosequence);
    splittosequence_param1.axis = 32;
    splittosequence_param1.keepdims = 44;
    MAKE_BUFFER(SplitToSequenceParam, splittosequence);
    int32_t axis = splittosequence_param1.axis;
    int32_t keepdims = splittosequence_param1.keepdims;
    EXPECT_EQ(32, axis);
    EXPECT_EQ(44, keepdims);
}