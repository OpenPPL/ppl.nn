#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/roialign.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_roialign) {
    DEFINE_ARG(RoiAlignParam, roialign);
    roialign_param1.mode = 1;
    roialign_param1.output_height = 32;
    roialign_param1.output_width = 33;
    roialign_param1.sampling_ratio = 4;
    roialign_param1.spatial_scale = 0.44;
    MAKE_BUFFER(RoiAlignParam, roialign);
    uint32_t mode = roialign_param3.mode;
    uint32_t output_height = roialign_param3.output_height;
    uint32_t output_width = roialign_param3.output_width;
    uint32_t sampling_ratio = roialign_param3.sampling_ratio;
    float spatial_scale = roialign_param3.spatial_scale;
    EXPECT_EQ(1, mode);
    EXPECT_EQ(32, output_height);
    EXPECT_EQ(33, output_width);
    EXPECT_EQ(4, sampling_ratio);
    EXPECT_FLOAT_EQ(0.44, spatial_scale);
}