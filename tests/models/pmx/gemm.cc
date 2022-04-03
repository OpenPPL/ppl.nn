#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/gemm.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_gemm) {
    DEFINE_ARG(GemmParam, gemm);
    gemm_param1.alpha = 0.21;
    gemm_param1.beta = 0.34;
    gemm_param1.transA = 32;
    gemm_param1.transB = 34;
    MAKE_BUFFER(GemmParam, gemm);
    float alpha = gemm_param3.alpha;
    float beta = gemm_param3.beta;
    int32_t transA = gemm_param3.transA;
    int32_t transB = gemm_param3.transB;
    EXPECT_FLOAT_EQ(0.21, alpha);
    EXPECT_FLOAT_EQ(0.34, beta);
    EXPECT_EQ(32, transA);
    EXPECT_EQ(34, transB);
}