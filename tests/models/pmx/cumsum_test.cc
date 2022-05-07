#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/cumsum.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_cumsum) {
    DEFINE_ARG(CumSumParam, cumsum);
    cumsum_param1.exclusive = 2;
    cumsum_param1.reverse = 3;
    MAKE_BUFFER(CumSumParam, cumsum);
    int32_t exclusive = cumsum_param3.exclusive;
    int32_t reverse = cumsum_param3.reverse;
    EXPECT_EQ(2, exclusive);
    EXPECT_EQ(3, reverse);
}