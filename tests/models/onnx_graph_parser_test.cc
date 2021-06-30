#include "ppl/nn/models/onnx/graph_parser.h"
#include "gtest/gtest.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include <string>
#include <iostream>

using namespace std;

class GraphParserTest : public testing::Test {};

TEST_F(GraphParserTest, Parse_Test) {
    const string onnx_file = PPLNN_TESTDATA_DIR + string("/conv.onnx");
    FILE* fp = fopen(onnx_file.c_str(), "r");
    if (!fp) {
        cout << "open onnx model file [" << onnx_file << "] failed!" << endl;
        return;
    }
    int fd = fileno(fp);
    google::protobuf::io::FileInputStream fis(fd);
    google::protobuf::io::CodedInputStream cis(&fis);
    cis.SetTotalBytesLimit(INT_MAX, INT_MAX);
    ::onnx::ModelProto pb_model;
    if (!pb_model.ParseFromCodedStream(&cis)) {
        return;
    }
    fclose(fp);
    ppl::nn::onnx::GraphParser graph_parser;
    ppl::nn::ir::Graph graph;
    auto status = graph_parser.Parse(pb_model.graph(), &graph);
    EXPECT_EQ(status, ppl::common::RC_SUCCESS);
}
