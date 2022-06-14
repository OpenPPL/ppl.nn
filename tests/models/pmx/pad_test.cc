#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/pad.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_pad) {
    DEFINE_ARG(PadParam, pad);
    pad_param1.mode = ppl::nn::onnx::PadParam::PAD_MODE_REFLECT;
    MAKE_BUFFER(PadParam, pad);
    int32_t mode = pad_param3.mode;
    EXPECT_EQ(ppl::nn::onnx::PadParam::PAD_MODE_REFLECT, mode);
}
