#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/argmax.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_argmax) {
    DEFINE_ARG(ArgMaxParam, arg_max);
    arg_max_param1.axis = 2;
    arg_max_param1.keepdims = 3;
    MAKE_BUFFER(ArgMaxParam, arg_max);
    int32_t axis = arg_max_param3.axis;
    int32_t keepdims = arg_max_param3.keepdims;
    EXPECT_EQ(2, axis);
    EXPECT_EQ(3, keepdims);
}