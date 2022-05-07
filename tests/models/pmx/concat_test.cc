#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/concat.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_concat) {
    DEFINE_ARG(ConcatParam, concat);
    concat_param1.axis = 2;
    MAKE_BUFFER(ConcatParam, concat);
    int32_t axis = concat_param3.axis;
    EXPECT_EQ(2, axis);
}