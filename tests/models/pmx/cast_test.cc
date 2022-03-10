#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/cast.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_cast) {
    DEFINE_ARG(CastParam, cast);
    cast_param1.to = 2;
    MAKE_BUFFER(CastParam, cast);
    int32_t to = cast_param3.to;
    EXPECT_EQ(2, to);
}