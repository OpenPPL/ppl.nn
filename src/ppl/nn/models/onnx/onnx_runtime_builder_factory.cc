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

OnnxRuntimeBuilder* OnnxRuntimeBuilderFactory::Create(const char* model_file, vector<unique_ptr<Engine>>&& engines) {
    FileMapping fm;
    if (fm.Init(model_file) != RC_SUCCESS) {
        LOG(ERROR) << "Init filemapping from file [" << model_file << "] error.";
        return nullptr;
    }
    return OnnxRuntimeBuilderFactory::Create(fm.Data(), fm.Size(), std::move(engines));
}

OnnxRuntimeBuilder* OnnxRuntimeBuilderFactory::Create(const char* model_buf, uint64_t buf_len,
                                                      vector<unique_ptr<Engine>>&& engines) {
    set<string> engine_names;
    for (auto e = engines.begin(); e != engines.end(); ++e) {
        auto ret_pair = engine_names.insert(e->get()->GetName());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated engine[" << e->get()->GetName() << "]";
            return nullptr;
        }
    }

    vector<unique_ptr<EngineImpl>> engine_impls;
    engine_impls.reserve(engines.size());
    for (auto e = engines.begin(); e != engines.end(); ++e) {
        auto impl = unique_ptr<EngineImpl>(static_cast<EngineImpl*>(e->release()));
        engine_impls.emplace_back(std::move(impl));
    }

    auto builder = new onnx::RuntimeBuilderImpl();
    if (builder) {
        auto status = builder->Init(model_buf, buf_len, std::move(engine_impls));
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "init OnnxRuntimeBuilder failed: " << GetRetCodeStr(status);
            delete builder;
            return nullptr;
        }
    }

    return builder;
}

}} // namespace ppl::nn
