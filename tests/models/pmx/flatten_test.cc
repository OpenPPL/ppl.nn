#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/flatten.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_flatten) {
    DEFINE_ARG(FlattenParam, flatten);
    flatten_param1.axis = 32;
    MAKE_BUFFER(FlattenParam, flatten);
    int32_t axis = flatten_param3.axis;
    EXPECT_EQ(32, axis);
}