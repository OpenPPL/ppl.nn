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

#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/models/onnx/runtime_builder_impl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/common/file_mapping.h"
#include <set>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RuntimeBuilder* OnnxRuntimeBuilderFactory::Create(const char* model_file, Engine** engines, uint32_t engine_num) {
    FileMapping fm;
    if (fm.Init(model_file) != RC_SUCCESS) {
        LOG(ERROR) << "Init filemapping from file [" << model_file << "] error.";
        return nullptr;
    }
    return OnnxRuntimeBuilderFactory::Create(fm.Data(), fm.Size(), engines, engine_num);
}

RuntimeBuilder* OnnxRuntimeBuilderFactory::Create(const char* model_buf, uint64_t buf_len, Engine** engines,
                                                  uint32_t engine_num) {
    set<string> engine_names;
    for (uint32_t i = 0; i < engine_num; ++i) {
        auto e = engines[i];
        auto ret_pair = engine_names.insert(e->GetName());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated engine[" << e->GetName() << "]";
            return nullptr;
        }
    }

    vector<EngineImpl*> engine_impls(engine_num);
    for (uint32_t i = 0; i < engine_num; ++i) {
        engine_impls[i] = static_cast<EngineImpl*>(engines[i]);
    }

    auto builder = new onnx::RuntimeBuilderImpl();
    if (builder) {
        auto status = builder->Init(model_buf, buf_len, std::move(engine_impls));
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "init RuntimeBuilder failed: " << GetRetCodeStr(status);
            delete builder;
            return nullptr;
        }
    }

    return builder;
}

}} // namespace ppl::nn
