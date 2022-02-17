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

#ifndef _ST_HPC_PPL_NN_MODELS_PMX_PMX_RUNTIME_BUILDER_H_
#define _ST_HPC_PPL_NN_MODELS_PMX_PMX_RUNTIME_BUILDER_H_

#include "ppl/nn/common/common.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/runtime/runtime.h"

namespace ppl { namespace nn {

class PPLNN_PUBLIC PmxRuntimeBuilder {
public:
    virtual ~PmxRuntimeBuilder() {}

    /**
       @brief init from a model file
       @param engines used to process this model
       @note engines are managed by the caller
    */
    virtual ppl::common::RetCode Init(const char* model_file, Engine** engines, uint32_t engine_num) = 0;

    /**
       @brief init from a model buffer
       @param engines used to process this model
       @note engines are managed by the caller
    */
    virtual ppl::common::RetCode Init(const char* model_buf, uint64_t buf_len, Engine** engines,
                                      uint32_t engine_num) = 0;

    /** @brief creates a Runtime instance */
    virtual Runtime* CreateRuntime() = 0;

    virtual ppl::common::RetCode Serialize(const char* output_file, const char* fmt) const = 0;
};

}} // namespace ppl::nn

#endif
