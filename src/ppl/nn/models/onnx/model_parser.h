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

#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_MODEL_PARSER_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_MODEL_PARSER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/models/onnx/model.h"

namespace ppl { namespace nn { namespace onnx {

class ModelParser final {
public:
    static ppl::common::RetCode Parse(const char* model_buf, uint64_t buf_len, const char* model_file_dir,
                                      Model* model);
    static ppl::common::RetCode Parse(const char* model_buf, uint64_t buf_len, const char* model_file_dir,
                                      const char** inputs, uint32_t nr_input, const char** outputs, uint32_t nr_output,
                                      Model* model);
};

}}} // namespace ppl::nn::onnx

#endif
