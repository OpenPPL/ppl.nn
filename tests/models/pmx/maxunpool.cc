#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/maxunpool.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_maxunpool) {
    DEFINE_ARG(MaxUnpoolParam, maxunpool);
    maxunpool_param1.kernel_shape = {3};
    maxunpool_param1.pads = {5};
    maxunpool_param1.strides = {1};
    MAKE_BUFFER(MaxUnpoolParam, maxunpool);
    std::vector<int32_t> kernel_shape = maxunpool_param3.kernel_shape;
    std::vector<int32_t> pads = maxunpool_param3.pads;
    std::vector<int32_t> strides = maxunpool_param3.strides;
    EXPECT_EQ(3, kernel_shape[0]);
    EXPECT_EQ(5, pads[0]);
    EXPECT_EQ(1, strides[0]);
}