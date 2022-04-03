#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/batch_normalization.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_batchnorm) {
    DEFINE_ARG(BatchNormalizationParam, bn);
    bn_param1.epsilon = 2.2;
    bn_param1.momentum = 3.3;
    MAKE_BUFFER(BatchNormalizationParam, bn);
    float epsilon = bn_param3.epsilon;
    float momentum = bn_param3.momentum;
    EXPECT_FLOAT_EQ(2.2, epsilon);
    EXPECT_FLOAT_EQ(3.3, momentum);
}
