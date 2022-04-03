#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/transpose.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_transpose) {
    DEFINE_ARG(TransposeParam, transpose);
    transpose_param1.perm = {56};
    MAKE_BUFFER(TransposeParam, transpose);
    std::vector<int32_t> perm = transpose_param3.perm;
    EXPECT_EQ(56, perm[0]);
}