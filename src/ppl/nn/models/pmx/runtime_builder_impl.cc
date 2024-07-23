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
#include "ppl/common/mmap.h"
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

RetCode RuntimeBuilderImpl::LoadModel(const char* model_buf, uint64_t buf_len, const Resources& resources,
                                      const LoadModelOptions& opt) {
    RetCode status;

    if (opt.external_data_dir && opt.external_data_dir[0] != '\0') {
        if (opt.external_buffer) {
            LOG(ERROR) << "only one of `external_data_dir` and `external_buffer` can be set.";
            return RC_INVALID_VALUE;
        }
    }

    auto fb_model = pmx::GetModel(model_buf);
    if (!fb_model) {
        LOG(ERROR) << "parse ppl model failed.";
        return RC_OTHER_ERROR;
    }

    LOG(INFO) << "ppl model version [" << fb_model->version() << "] created by [" << fb_model->producer()->c_str()
              << "]";

    SetResources(resources, &resource_);

    vector<EngineImpl*> seq2engine;
    status = ParseEngines(fb_model->engines(), resource_.engines, &seq2engine);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseEngines failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GraphParser::Parse(fb_model->graph(), seq2engine, opt, topo_.get(), graph_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::LoadModel(const char* model_file, const Resources& resources, const LoadModelOptions& opt) {
    if (opt.external_data_dir && opt.external_data_dir[0] != '\0') {
        if (opt.external_buffer && opt.external_buffer_size > 0) {
            LOG(ERROR) << "only one of `external_data_dir` and `external_buffer` can be set.";
            return RC_INVALID_VALUE;
        }
    }

    Mmap fm;
    auto status = fm.Init(model_file, Mmap::READ);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "mapping file [" << model_file << "] faild.";
        return status;
    }

    string model_dir;
    LoadModelOptions new_opt;

    const LoadModelOptions* opt_ptr;
    if (opt.external_data_dir && opt.external_data_dir[0] != '\0') {
        opt_ptr = &opt;
    } else if (opt.external_buffer) {
        opt_ptr = &opt;
    } else {
        new_opt = opt;
        opt_ptr = &new_opt;

        auto pos = string(model_file).find_last_of("/\\");
        if (pos == string::npos) {
            model_dir = ".";
        } else {
            model_dir.assign(model_file, pos);
        }
        new_opt.external_data_dir = model_dir.c_str();
    }

    return LoadModel(fm.GetData(), fm.GetSize(), resources, *opt_ptr);
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

RetCode RuntimeBuilderImpl::Serialize(const char* fmt, const void* options, utils::DataStream* ds) const {
    if (fmt == string("pmx")) {
        const pmx::SaveModelOptions default_opt;
        const pmx::SaveModelOptions* opt;
        if (options) {
            opt = (const pmx::SaveModelOptions*)options;
            if ((opt->external_data_dir && opt->external_data_dir[0] != '\0') &&
                (opt->external_data_file && opt->external_data_file[0] != '\0')) {
                LOG(ERROR) << "only one of `external_data_dir` and `external_data_file` can be set.";
                return RC_INVALID_VALUE;
            }
        } else {
            opt = &default_opt;
        }

        pmx::Serializer serializer;
        return serializer.Serialize(*opt, topo_.get(), resource_.engines, *graph_info_, ds);
    }

    LOG(ERROR) << "model format[" << fmt << "] is not supported.";
    return RC_UNSUPPORTED;
}

RetCode RuntimeBuilderImpl::ReserveTensor(const char* tensor_name) {
    auto edge = topo_->GetEdge(tensor_name);
    if (!edge) {
        LOG(ERROR) << "ReserveTensor: cannot find tensor named[" << tensor_name << "]";
        return RC_NOT_FOUND;
    }

    resource_.reserved_edgeids.insert(edge->GetId());
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
