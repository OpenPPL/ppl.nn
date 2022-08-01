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

#ifndef _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_PARSER_H_
#define _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_PARSER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/quantization/quant_param_info.h"

namespace ppl { namespace nn {

class QuantParamParser final {
public:
    static ppl::common::RetCode ParseFile(const char* json_file, QuantParamInfo*);
    static ppl::common::RetCode ParseBuffer(const char* json_buf, uint64_t size, QuantParamInfo*);

private:
    QuantParamParser() = delete;
};

}} // namespace ppl::nn

#endif
