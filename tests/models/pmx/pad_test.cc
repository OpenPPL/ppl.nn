#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/pad.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_pad) {
    DEFINE_ARG(PadParam, pad);
    pad_param1.mode = ppl::nn::common::PadParam::PAD_MODE_REFLECT;
    MAKE_BUFFER(PadParam, pad);
    ppl::nn::common::PadParam::pad_mode_t mode = pad_param3.mode;
    EXPECT_EQ(ppl::nn::common::PadParam::PAD_MODE_REFLECT, mode);
}