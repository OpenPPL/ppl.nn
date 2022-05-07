#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/non_max_suppression.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_nonmaxsuppression) {
    DEFINE_ARG(NonMaxSuppressionParam, nonmaxsuppression);
    nonmaxsuppression_param1.center_point_box = 3;
    MAKE_BUFFER(NonMaxSuppressionParam, nonmaxsuppression);
    int32_t center_point_box = nonmaxsuppression_param3.center_point_box;
    EXPECT_EQ(3, center_point_box);
}
