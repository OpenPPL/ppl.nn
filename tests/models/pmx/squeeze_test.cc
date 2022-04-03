#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/squeeze.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_squeeze) {
    DEFINE_ARG(SqueezeParam, squeeze);
    squeeze_param1.axes = {32};
    MAKE_BUFFER(SqueezeParam, squeeze);
    std::vector<int32_t> axes = squeeze_param3.axes;
    EXPECT_EQ(32, axes[0]);
}