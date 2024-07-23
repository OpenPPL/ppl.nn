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

#ifndef _ST_HPC_PPL_NN_MODELS_PMX_LOAD_MODEL_OPTIONS_H_
#define _ST_HPC_PPL_NN_MODELS_PMX_LOAD_MODEL_OPTIONS_H_

namespace ppl { namespace nn { namespace pmx {

#include "ppl/nn/common/common.h"

struct PPLNN_PUBLIC LoadModelOptions final {
    /** load constant from external data files in `external_data_dir` if `EXTERNAL_MULTI_FILES` is enabled. */
    const char* external_data_dir = nullptr;

    struct {
        const char* external_buffer = nullptr;
        uint64_t external_buffer_size = 0;
    };
};

}}} // namespace ppl::nn::pmx

#endif
