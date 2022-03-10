#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/resize.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_resize) {
    DEFINE_ARG(ResizeParam, resize);
    resize_param1.coord_trans_mode = ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_ASYMMETRIC;
    resize_param1.cubic_coeff_a = 0.33;
    resize_param1.exclude_outside = 23;
    resize_param1.extrapolation_value = 0.45;
    resize_param1.mode = ppl::nn::common::ResizeParam::RESIZE_MODE_LINEAR;
    resize_param1.nearest_mode = ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL;
    MAKE_BUFFER(ResizeParam, resize);
    ppl::nn::common::ResizeParam::resize_coord_trans_mode_t coord_trans_mode = resize_param3.coord_trans_mode;
    float cubic_coeff_a = resize_param3.cubic_coeff_a;
    int32_t exclude_outside = resize_param3.exclude_outside;
    float extrapolation_value = resize_param3.extrapolation_value;
    ppl::nn::common::ResizeParam::resize_mode_t mode = resize_param3.mode;
    ppl::nn::common::ResizeParam::resize_nearest_mode_t nearest_mode = resize_param3.nearest_mode;
    EXPECT_EQ(ppl::nn::common::ResizeParam::RESIZE_COORD_TRANS_MODE_ASYMMETRIC, coord_trans_mode);
    EXPECT_FLOAT_EQ(0.33, cubic_coeff_a);
    EXPECT_EQ(23, exclude_outside);
    EXPECT_FLOAT_EQ(0.45, extrapolation_value);
    EXPECT_EQ(ppl::nn::common::ResizeParam::RESIZE_MODE_LINEAR, mode);
    EXPECT_EQ(ppl::nn::common::ResizeParam::RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL, nearest_mode);
}