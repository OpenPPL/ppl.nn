#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/topk.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_topk) {
    DEFINE_ARG(TopKParam, topk);
    topk_param1.axis = 32;
    topk_param1.largest = 23;
    topk_param1.sorted = 11;
    MAKE_BUFFER(TopKParam, topk);
    int32_t axis = topk_param3.axis;
    int32_t largest = topk_param3.largest;
    int32_t sorted = topk_param3.sorted;
    EXPECT_EQ(32, axis);
    EXPECT_EQ(23, largest);
    EXPECT_EQ(11, sorted);
}