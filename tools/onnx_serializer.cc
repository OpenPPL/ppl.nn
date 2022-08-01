// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/common/file_mapping.h"
#include "ppl/nn/models/onnx/model_parser.h"
#include "ppl/nn/models/onnx/serializer.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(ERROR) << "usage: " << argv[0] << " input_onnx_model output_onnx_model";
        return -1;
    }

    const string input_file(argv[1]);
    const string output_file(argv[2]);

    FileMapping fm;
    auto status = fm.Init(input_file.c_str(), FileMapping::READ);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "open input model[" << input_file << "] failed.";
        return -1;
    }

    string parent_dir;
    auto pos = string(input_file).find_last_of("/\\");
    if (pos == string::npos) {
        parent_dir = ".";
    } else {
        parent_dir.assign(input_file.c_str(), pos);
    }

    Model model;
    status = ModelParser::Parse(fm.GetData(), fm.GetSize(), parent_dir.c_str(), &model);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse model[" << input_file << "] failed.";
        return -1;
    }

    Serializer serializer;
    status = serializer.Serialize(output_file.c_str(), model);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "serialize model to [" << output_file << "] failed.";
        return -1;
    }

    return 0;
}
