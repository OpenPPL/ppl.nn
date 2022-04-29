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
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/ir/full_graph_topo.h"
#include "ppl/nn/models/pmx/runtime_builder_impl.h"
#include "ppl/nn/models/pmx/graph_parser.h"
#include "ppl/nn/models/pmx/pmx_serializer.h"
using namespace std;
using namespace ppl::common;
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx {

RuntimeBuilderImpl::RuntimeBuilderImpl() {
    topo_ = make_shared<ir::FullGraphTopo>();
    graph_info_ = make_shared<RuntimeGraphInfo>();
    aux_info_ = make_shared<RuntimeAuxInfo>();
}

RuntimeBuilderImpl::~RuntimeBuilderImpl() {
    aux_info_.reset();
    graph_info_.reset();
    topo_.reset();
}

static RetCode ParseEngines(const Vector<Offset<Engine>>* fb_engines, const vector<EngineImpl*>& engines,
                            vector<EngineImpl*>* seq2engine) {
    seq2engine->reserve(engines.size());
    for (auto x = fb_engines->begin(); x != fb_engines->end(); ++x) {
        bool found = false;
        for (auto y = engines.begin(); y != engines.end(); ++y) {
            auto engine = *y;
            if (strcmp(engine->GetName(), x->name()->c_str()) == 0) {
                if (x->data()->size() > 0) {
                    auto status = engine->DeserializeData(x->data()->data(), x->data()->size());
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "DeserializeData for engine[" << engine->GetName()
                                   << "]: " << GetRetCodeStr(status);
                        return status;
                    }
                }
                seq2engine->push_back(engine);
                found = true;
                break;
            }
        }
        if (!found) {
            LOG(ERROR) << "cannot find engine[" << x->name()->c_str() << "]";
            return RC_NOT_FOUND;
        }
    }

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::Init(const char* model_buf, uint64_t buf_len, ppl::nn::Engine** engines,
                                 uint32_t engine_num) {
    RetCode status;

    resource_.engines.resize(engine_num);
    for (uint32_t i = 0; i < engine_num; ++i) {
        resource_.engines[i] = static_cast<EngineImpl*>(engines[i]);
    }

    auto fb_model = pmx::GetModel(model_buf);
    if (!fb_model) {
        LOG(ERROR) << "parse ppl model failed.";
        return RC_OTHER_ERROR;
    }

    LOG(INFO) << "ppl model version: " << fb_model->version();

    vector<EngineImpl*> seq2engine;
    status = ParseEngines(fb_model->engines(), resource_.engines, &seq2engine);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseEngines failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GraphParser::Parse(fb_model->graph(), seq2engine, topo_.get(), graph_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(status);
        return status;
    }

    partial_runtime_creator_.Init(topo_.get(), graph_info_, &init_info_.name2nodeid);

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::Init(const char* model_file, ppl::nn::Engine** engines, uint32_t engine_num) {
    FileMapping fm;
    auto status = fm.Init(model_file);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init filemapping from file [" << model_file << "] faild: " << GetRetCodeStr(status);
        return status;
    }
    return Init(fm.Data(), fm.Size(), engines, engine_num);
}

RetCode RuntimeBuilderImpl::Preprocess() {
    auto status = aux_info_->Init(topo_.get(), resource_.reserved_edgeids);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    status = init_info_.Init(topo_.get());
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

    auto status = runtime->Init(topo_, graph_info_, aux_info_, init_info_, resource_.reserved_edgeids);
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
    if (fmt != string("pmx")) {
        LOG(ERROR) << "model format[" << fmt << "] is not supported.";
        return RC_UNSUPPORTED;
    }

    pmx::PmxSerializer serializer;
    return serializer.Serialize(output_file, topo_.get(), resource_.engines, *graph_info_);
}

/* -------------------------------------------------------------------------- */

RetCode RuntimeBuilderImpl::ReserveTensor(RuntimeBuilderImpl* impl, va_list args) {
    auto tensor_name = va_arg(args, const char*);

    auto edge = impl->topo_->GetEdge(tensor_name);
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
    if (option >= PRB_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << PRB_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::pmx
