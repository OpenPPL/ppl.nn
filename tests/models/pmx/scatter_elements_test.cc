#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/scatter_elements.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_scatterelements) {
    DEFINE_ARG(ScatterElementsParam, scatterelements);
    scatterelements_param1.axis = 32;
    MAKE_BUFFER(ScatterElementsParam, scatterelements);
    int32_t axis = scatterelements_param1.axis;
    EXPECT_EQ(32, axis);
}
