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

#include <stdarg.h>
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
    graph_info_ = make_shared<RuntimeGraphInfo>();
    aux_info_ = make_shared<RuntimeAuxInfo>();
}

RuntimeBuilderImpl::~RuntimeBuilderImpl() {
    aux_info_.reset();
    graph_info_.reset();
}

RetCode RuntimeBuilderImpl::LoadModel(const char* model_buf, uint64_t buf_len, const char* model_file_dir) {
    auto status = ModelParser::Parse(model_buf, buf_len, model_file_dir, &graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(status);
        return status;
    }

    partial_runtime_creator_.Init(graph_.topo.get(), graph_info_, &init_info_.name2nodeid);

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::LoadModel(const char* model_file) {
    FileMapping fm;
    auto status = fm.Init(model_file);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init filemapping from file [" << model_file << "] faild: " << GetRetCodeStr(status);
        return status;
    }

    string parent_dir;
    auto pos = string(model_file).find_last_of("/\\");
    if (pos == string::npos) {
        parent_dir = ".";
    } else {
        parent_dir.assign(model_file, pos);
    }

    return LoadModel(fm.Data(), fm.Size(), parent_dir.c_str());
}

RetCode RuntimeBuilderImpl::SetResources(const Resources& resource) {
    resource_.engines.resize(resource.engine_num);
    for (uint32_t i = 0; i < resource.engine_num; ++i) {
        resource_.engines[i] = static_cast<EngineImpl*>(resource.engines[i]);
    }

    resource_.graph_partitioner = make_shared<EngineGraphPartitioner>();
    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::Preprocess() {
    auto status = utils::ProcessGraph(resource_, &graph_, graph_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "process graph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = aux_info_->Init(graph_.topo.get(), resource_.reserved_edgeids);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    status = init_info_.Init(graph_.topo.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeInitInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

Runtime* RuntimeBuilderImpl::CreateRuntime() {
    auto runtime = new RuntimeImpl();
    if (!runtime) {
        return nullptr;
    }

    auto status = runtime->Init(graph_.topo, graph_info_, aux_info_, init_info_, resource_.reserved_edgeids);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init runtime failed: " << GetRetCodeStr(status);
        delete runtime;
        return nullptr;
    }

    return runtime;
}

Runtime* RuntimeBuilderImpl::CreateRuntime(const char** begin_ops, uint32_t begin_op_num, const char** end_ops,
                                           uint32_t end_op_num) {
    return partial_runtime_creator_.Create(begin_ops, begin_op_num, end_ops, end_op_num, resource_.reserved_edgeids);
}

RetCode RuntimeBuilderImpl::Serialize(const char* output_file, const char* fmt) const {
#ifdef PPLNN_ENABLE_PMX_MODEL
    if (fmt != string("pmx")) {
        LOG(ERROR) << "model format[" << fmt << "] is not supported.";
        return RC_UNSUPPORTED;
    }

    pmx::PmxSerializer serializer;
    return serializer.Serialize(output_file, graph_.topo.get(), resource_.engines, *graph_info_);
#else
    LOG(ERROR) << "model format[" << fmt << "] is not supported.";
    return RC_UNSUPPORTED;
#endif
}

/* -------------------------------------------------------------------------- */

RetCode RuntimeBuilderImpl::ReserveTensor(RuntimeBuilderImpl* impl, va_list args) {
    auto tensor_name = va_arg(args, const char*);

    auto edge = impl->graph_.topo->GetEdge(tensor_name);
    if (!edge) {
        LOG(ERROR) << "ReserveTensor: cannot find tensor named[" << tensor_name << "]";
        return RC_NOT_FOUND;
    }

    impl->resource_.reserved_edgeids.insert(edge->GetId());
    return RC_SUCCESS;
}

RuntimeBuilderImpl::ConfHandlerFunc RuntimeBuilderImpl::conf_handlers_[] = {
    RuntimeBuilderImpl::ReserveTensor,
};

RetCode RuntimeBuilderImpl::Configure(uint32_t option, ...) {
    if (option >= ORB_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << ORB_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::onnx
