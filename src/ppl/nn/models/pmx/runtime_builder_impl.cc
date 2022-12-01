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
#include "ppl/nn/models/pmx/serializer.h"
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

static void SetResources(const RuntimeBuilder::Resources& src, utils::SharedResource* dst) {
    dst->engines.resize(src.engine_num);
    for (uint32_t i = 0; i < src.engine_num; ++i) {
        dst->engines[i] = static_cast<EngineImpl*>(src.engines[i]);
    }
}

RetCode RuntimeBuilderImpl::LoadModel(const char* model_buf, uint64_t buf_len, const Resources& resources) {
    RetCode status;

    auto fb_model = pmx::GetModel(model_buf);
    if (!fb_model) {
        LOG(ERROR) << "parse ppl model failed.";
        return RC_OTHER_ERROR;
    }

    LOG(INFO) << "ppl model version: " << fb_model->version();

    SetResources(resources, &resource_);

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

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::LoadModel(const char* model_file, const Resources& resources) {
    FileMapping fm;
    auto status = fm.Init(model_file, FileMapping::READ);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "mapping file [" << model_file << "] faild: " << fm.GetErrorMessage();
        return status;
    }
    return LoadModel(fm.GetData(), fm.GetSize(), resources);
}

RetCode RuntimeBuilderImpl::Preprocess() {
    auto status = aux_info_->Init(topo_.get(), resource_.reserved_edgeids);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

Runtime* RuntimeBuilderImpl::CreateRuntime() const {
    auto runtime = new RuntimeImpl();
    if (!runtime) {
        return nullptr;
    }

    auto status = runtime->Init(topo_, graph_info_, aux_info_, resource_.reserved_edgeids);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init runtime failed: " << GetRetCodeStr(status);
        delete runtime;
        return nullptr;
    }

    return runtime;
}

RetCode RuntimeBuilderImpl::Serialize(const char* fmt, utils::DataStream* ds) const {
    if (fmt == string("pmx")) {
        pmx::Serializer serializer;
        return serializer.Serialize(topo_.get(), resource_.engines, *graph_info_, ds);
    }

    LOG(ERROR) << "model format[" << fmt << "] is not supported.";
    return RC_UNSUPPORTED;
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
        LOG(ERROR) << "invalid option[" << option << "] >= [" << (uint32_t)PRB_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::pmx
