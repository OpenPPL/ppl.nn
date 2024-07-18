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

#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_RUNTIME_BUILDER_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_RUNTIME_BUILDER_H_

#include "ppl/nn/common/common.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/utils/data_stream.h"

namespace ppl { namespace nn { namespace onnx {

class PPLNN_PUBLIC RuntimeBuilder {
public:
    struct PPLNN_PUBLIC Resources final {
        /** engines are managed by the caller */
        Engine** engines = nullptr;
        uint32_t engine_num = 0;
    };

public:
    virtual ~RuntimeBuilder() {}

    /** @brief load model from a file */
    virtual ppl::common::RetCode LoadModel(const char* model_file) = 0;

    /**
       @brief load model from a buffer
       @param model_file_dir used to parse external data. can be nullptr if no external data.
    */
    virtual ppl::common::RetCode LoadModel(const char* model_buf, uint64_t buf_len,
                                           const char* model_file_dir = nullptr) = 0;

    /** @brief load partial model from a file */
    virtual ppl::common::RetCode LoadModel(const char* model_file, const char** inputs, uint32_t nr_input,
                                           const char** outputs, uint32_t nr_output) = 0;

    /**
       @brief load partial model from a buffer
       @param model_file_dir used to parse external data. can be nullptr if no external data.
    */
    virtual ppl::common::RetCode LoadModel(const char* model_buf, uint64_t buf_len, const char** inputs,
                                           uint32_t nr_input, const char** outputs, uint32_t nr_output,
                                           const char* model_file_dir = nullptr) = 0;

    /**
       @brief set resources for preprocessing and creating `Runtime`.
       MUST be called before `Preprocess()` and `CreateRuntime()`.
    */
    virtual ppl::common::RetCode SetResources(const Resources&) = 0;

    /** @brief mark a tensor as reserved in order to avoid being fused and reused duing inferencing */
    virtual ppl::common::RetCode ReserveTensor(const char* tensor_name) = 0;

    /** @note MUST be called before `CreateRuntime()` */
    virtual ppl::common::RetCode Preprocess() = 0;

    /** @brief creates a `Runtime` instance. This function is thread-safe. */
    virtual Runtime* CreateRuntime() const = 0;

    virtual ppl::common::RetCode Serialize(const char* fmt, const void* options, ppl::nn::utils::DataStream*) const = 0;
};

}}} // namespace ppl::nn::onnx

#endif
