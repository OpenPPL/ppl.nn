#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/lrn.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_lrn) {
    DEFINE_ARG(LRNParam, lrn);
    lrn_param1.alpha = 0.32;
    lrn_param1.beta = 0.34;
    lrn_param1.bias = 0.24;
    lrn_param1.size = 12;
    MAKE_BUFFER(LRNParam, lrn);
    float alpha = lrn_param3.alpha;
    float beta = lrn_param3.beta;
    float bias = lrn_param3.bias;
    size_t size = lrn_param3.size;
    EXPECT_FLOAT_EQ(0.32, alpha);
    EXPECT_FLOAT_EQ(0.34, beta);
    EXPECT_FLOAT_EQ(0.24, bias);
    EXPECT_EQ(12, size);
}