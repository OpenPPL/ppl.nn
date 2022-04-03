#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/leaky_relu.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_leakyrelu) {
    DEFINE_ARG(LeakyReluParam, leakyrelu);
    leakyrelu_param1.alpha = 0.32;
    MAKE_BUFFER(LeakyReluParam, leakyrelu);
    float alpha = leakyrelu_param3.alpha;
    EXPECT_FLOAT_EQ(0.32, alpha);
}
