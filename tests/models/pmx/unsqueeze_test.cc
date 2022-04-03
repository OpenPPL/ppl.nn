#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/unsqueeze.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_unsqueeze) {
    DEFINE_ARG(UnsqueezeParam, unsqueeze);
    unsqueeze_param1.axes = {33};
    MAKE_BUFFER(UnsqueezeParam, unsqueeze);
    std::vector<int32_t> axes = unsqueeze_param3.axes;
    EXPECT_EQ(33, axes[0]);
}