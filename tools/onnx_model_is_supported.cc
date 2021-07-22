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
#include "ppl/nn/common/logger.h"
#include <iostream>
using namespace ppl::nn;
using namespace ppl::common;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "usage: " << argv[0] << " onnx-model" << endl;
        return 1;
    }
    const char* model_file = argv[1];

    FileMapping fm;
    auto status = fm.Init(model_file);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init filemapping from file [" << model_file << "] error: " << GetRetCodeStr(status);
        return 1;
    }

    ir::Graph graph;
    status = ppl::nn::onnx::ModelParser::Parse(fm.Data(), fm.Size(), &graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse model failed: " << GetRetCodeStr(status);
        return 1;
    }
    (void)graph;

    return 0;
}
