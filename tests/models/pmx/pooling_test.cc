#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/pooling.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_pooling) {
    DEFINE_ARG(PoolingParam, pooling);
    pooling_param1.kernel_shape = {3};
    pooling_param1.dilations = {4};
    pooling_param1.strides = {5};
    pooling_param1.pads = {6};
    pooling_param1.ceil_mode = 1;
    MAKE_BUFFER(PoolingParam, pooling);
    std::vector<int32_t> kernel_shape = pooling_param3.kernel_shape;
    std::vector<int32_t> dilations = pooling_param3.dilations;
    std::vector<int32_t> strides = pooling_param3.strides;
    std::vector<int32_t> pads = pooling_param3.pads;
    int32_t ceil_mode = pooling_param3.ceil_mode;
    EXPECT_EQ(3, kernel_shape[0]);
    EXPECT_EQ(4, dilations[0]);
    EXPECT_EQ(5, strides[0]);
    EXPECT_EQ(6, pads[0]);
    EXPECT_EQ(1, ceil_mode);
}