#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/gathernd.h"

using namespace std;
using namespace ppl::nn::common;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_gathernd) {
    DEFINE_ARG(GatherNDParam, gathernd);
    gathernd_param1.batch_dims = 32;
    MAKE_BUFFER(GatherNDParam, gathernd);
    int32_t batch_dims = gathernd_param3.batch_dims;
    EXPECT_EQ(32, batch_dims);
}