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
#include "ppl/nn/common/logger.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/optimizers/engine_graph_partitioner.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/models/onnx/model_parser.h"
#include "ppl/nn/models/onnx/runtime_builder_impl.h"
using namespace std;
using namespace ppl::common;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/pmx_serializer.h"
#endif

namespace ppl { namespace nn { namespace onnx {

RuntimeBuilderImpl::RuntimeBuilderImpl() {
    resource_ = make_shared<utils::SharedResource>();
    graph_info_ = make_shared<RuntimeGraphInfo>();
    aux_info_ = make_shared<RuntimeAuxInfo>();
}

RuntimeBuilderImpl::~RuntimeBuilderImpl() {
    graph_.topo.reset();
    graph_.data.reset();
    aux_info_.reset();
    graph_info_.reset();
    resource_.reset();
}

RetCode RuntimeBuilderImpl::Init(const char* model_buf, uint64_t buf_len, Engine** engines, uint32_t engine_num) {
    resource_->engines.resize(engine_num);
    for (uint32_t i = 0; i < engine_num; ++i) {
        resource_->engines[i] = static_cast<EngineImpl*>(engines[i]);
    }

    resource_->graph_partitioner = make_shared<EngineGraphPartitioner>();

    auto status = ModelParser::Parse(model_buf, buf_len, &graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::Init(const char* model_file, Engine** engines, uint32_t engine_num) {
    FileMapping fm;
    auto status = fm.Init(model_file);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init filemapping from file [" << model_file << "] faild: " << GetRetCodeStr(status);
        return status;
    }
    return Init(fm.Data(), fm.Size(), engines, engine_num);
}

RetCode RuntimeBuilderImpl::Preprocess() {
    auto status = utils::ProcessGraph(resource_.get(), &graph_, graph_info_.get());
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

RetCode RuntimeBuilderImpl::Serialize(const char* output_file, const char* fmt) const {
#ifdef PPLNN_ENABLE_PMX_MODEL
    if (fmt != string("pmx")) {
        LOG(ERROR) << "model format[" << fmt << "] is not supported.";
        return RC_UNSUPPORTED;
    }

    pmx::PmxSerializer serializer;
    return serializer.Serialize(output_file, graph_.topo.get(), resource_->engines, *graph_info_);
#else
    LOG(ERROR) << "model format[" << fmt << "] is not supported.";
    return RC_UNSUPPORTED;
#endif
}

}}} // namespace ppl::nn::onnx
