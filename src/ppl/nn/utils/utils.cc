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

#include "ppl/nn/utils/utils.h"
#include <fstream>
#include <sstream>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

RetCode ReadFileContent(const char* fname, string* buf) {
    ifstream ifile;

    ifile.open(fname, ios_base::in);
    if (!ifile.is_open()) {
        LOG(ERROR) << "open file[" << fname << "] failed.";
        return RC_NOT_FOUND;
    }

    stringstream ss;
    ss << ifile.rdbuf();
    *buf = ss.str();

    ifile.close();
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::utils
