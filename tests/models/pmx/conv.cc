#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_conv) {
    DEFINE_ARG(ConvParam, conv);
    conv_param1.auto_pad = 2;
    conv_param1.group = 3;
    conv_param1.kernel_shape = {3};
    conv_param1.dilations = {1};
    conv_param1.strides = {0};
    conv_param1.pads = {1};
    MAKE_BUFFER(ConvParam, conv);
    uint32_t auto_pad = conv_param3.auto_pad;
    int32_t group = conv_param3.group;
    std::vector<int32_t> kernel_shape = conv_param3.kernel_shape;
    std::vector<int32_t> dilations = conv_param3.dilations;
    std::vector<int32_t> strides = conv_param3.strides;
    std::vector<int32_t> pads = conv_param3.pads;
    EXPECT_EQ(2, auto_pad);
    EXPECT_EQ(3, group);
    EXPECT_EQ(3, kernel_shape[0]);
    EXPECT_EQ(1, dilations[0]);
    EXPECT_EQ(0, strides[0]);
    EXPECT_EQ(1, pads[0]);
}