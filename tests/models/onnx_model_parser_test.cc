#include "ppl/nn/models/onnx/model_parser.h"
#include "gtest/gtest.h"
#include <string>

using namespace std;
using namespace ppl::nn;
class ModelParserTest : public testing::Test {};

TEST_F(ModelParserTest, TestModelParser) {
    ir::Graph graph;
    const string onnx_file = PPLNN_TESTDATA_DIR + string("/conv.onnx");
    auto res = onnx::ModelParser::Parse(onnx_file.c_str(), &graph);
    EXPECT_EQ(res, ppl::common::RC_SUCCESS);
}
