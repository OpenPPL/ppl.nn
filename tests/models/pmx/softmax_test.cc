#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/softmax.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_softmax) {
    DEFINE_ARG(SoftmaxParam, softmax);
    softmax_param1.axis = 32;
    MAKE_BUFFER(SoftmaxParam, softmax);
    int32_t axis = softmax_param3.axis;
    EXPECT_EQ(32, axis);
}