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

#include "ppl/nn/common/logger.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/models/onnx/model_parser.h"
#include "ppl/nn/models/onnx/runtime_builder_impl.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RuntimeBuilderImpl::RuntimeBuilderImpl() {
    resource_ = make_shared<utils::SharedResource>();
    graph_info_ = make_shared<RuntimeGraphInfo>();
    aux_info_ = make_shared<RuntimeAuxInfo>();
}

RuntimeBuilderImpl::~RuntimeBuilderImpl() {
    aux_info_.reset();
    graph_info_.reset();
    resource_.reset();
}

RetCode RuntimeBuilderImpl::Init(const char* model_buf, size_t buf_len, vector<EngineImpl*>&& engines) {
    resource_->engines = std::move(engines);

    auto status = ModelParser::Parse(model_buf, buf_len, &graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::ProcessGraph(resource_.get(), &graph_, graph_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "process graph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenerateRuntimeAuxInfo(graph_.topo.get(), aux_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

Runtime* RuntimeBuilderImpl::CreateRuntime() {
    auto runtime = new RuntimeImpl();
    if (!runtime) {
        return nullptr;
    }

    auto status = runtime->Init(graph_.topo, graph_info_, aux_info_, resource_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init runtime failed: " << GetRetCodeStr(status);
        delete runtime;
        return nullptr;
    }

    return runtime;
}

}}} // namespace ppl::nn::onnx
