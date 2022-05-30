#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/reduce.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_reduce) {
    DEFINE_ARG(ReduceParam, reduce);
    reduce_param1.keepdims = 3;
    reduce_param1.axes = {23};
    MAKE_BUFFER(ReduceParam, reduce);
    int32_t keepdims = reduce_param3.keepdims;
    std::vector<int32_t> axes = reduce_param3.axes;
    EXPECT_EQ(3, keepdims);
    EXPECT_EQ(23, axes[0]);
}
