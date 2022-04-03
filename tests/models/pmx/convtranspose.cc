#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv_transpose.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_convtranspose) {
    DEFINE_ARG(ConvTransposeParam, conv_trans)
    conv_trans_param1.auto_pad = "SAME_UPPER";
    conv_trans_param1.group = 3;
    conv_trans_param1.kernel_shape = {3};
    conv_trans_param1.dilations = {1};
    conv_trans_param1.strides = {0};
    conv_trans_param1.pads = {1};
    conv_trans_param1.output_padding = {0};
    conv_trans_param1.output_shape = {4};
    MAKE_BUFFER(ConvTransposeParam, conv_trans);
    string auto_pad = conv_trans_param3.auto_pad;
    int64_t group = conv_trans_param3.group;
    std::vector<int32_t> kernel_shape = conv_trans_param3.kernel_shape;
    std::vector<int32_t> dilations = conv_trans_param3.dilations;
    std::vector<int32_t> strides = conv_trans_param3.strides;
    std::vector<int32_t> pads = conv_trans_param3.pads;
    std::vector<int32_t> output_padding = conv_trans_param3.output_padding;
    std::vector<int32_t> output_shape = conv_trans_param3.output_shape;
    EXPECT_STREQ("SAME_UPPER", auto_pad.c_str());
    EXPECT_EQ(3, group);
    EXPECT_EQ(3, kernel_shape[0]);
    EXPECT_EQ(1, dilations[0]);
    EXPECT_EQ(0, strides[0]);
    EXPECT_EQ(1, pads[0]);
    EXPECT_EQ(0, output_padding[0]);
    EXPECT_EQ(4, output_shape[0]);
}
