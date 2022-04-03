#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/gather.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_gather) {
    DEFINE_ARG(GatherParam, gather);
    gather_param1.axis = 32;
    MAKE_BUFFER(GatherParam, gather);
    int32_t axis = gather_param3.axis;
    EXPECT_EQ(32, axis);
}