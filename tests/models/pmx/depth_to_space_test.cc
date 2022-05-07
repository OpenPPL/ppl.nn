#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/depth_to_space.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_depthtospace) {
    DEFINE_ARG(DepthToSpaceParam, depth);
    depth_param1.blocksize = 32;
    depth_param1.mode = 1;
    MAKE_BUFFER(DepthToSpaceParam, depth);
    int32_t blocksize = depth_param3.blocksize;
    int32_t mode = depth_param3.mode;
    EXPECT_EQ(32, blocksize);
    EXPECT_EQ(1, mode);
}
