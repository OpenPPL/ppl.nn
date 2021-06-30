#include "ppl/nn/quantization/quant_param_parser.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;

TEST(QuantParamParserTest, misc) {
    QuantParamParser parser;
    QuantParamInfo info;
    const string test_conf = PPLNN_TESTDATA_DIR + string("/ppq_test_qparams.json");
    auto status = parser.Parse(test_conf.c_str(), &info);
    EXPECT_EQ(RC_SUCCESS, status);

    auto item_iter = info.tensor_params.find("input.1");
    EXPECT_NE(info.tensor_params.end(), item_iter);
    auto field_iter = item_iter->second.fields.find("algorithm");
    EXPECT_NE(item_iter->second.fields.end(), field_iter);
    EXPECT_EQ("kl", field_iter->second.content);
}
