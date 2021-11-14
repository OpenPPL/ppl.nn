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

#include "ppl/nn/engines/cuda/engine.h"

#include <stdarg.h>

#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/engines/cuda/optimizer/opt_graph.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/module/op_compile_manager.h"
#include "ppl/nn/quantization/quant_param_parser.cc"
#include "ppl/nn/utils/array.h"
#include "rapidjson/document.h"
#include "rapidjson/error/error.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaEngine::Init(const CudaEngineOptions& options) {
    options_ = options;
    return device_.Init(options);
}

EngineContext* CudaEngine::CreateEngineContext() {
    auto ctx = unique_ptr<CudaEngineContext>(new CudaEngineContext());
    auto status = ctx->Init(options_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init CudaEngineContext failed: " << GetRetCodeStr(status);
        return nullptr;
    }

    return ctx.release();
}

bool CudaEngine::Supports(const ir::Node* node) const {
    auto& type = node->GetType();
    return (OptKernelCreatorManager::Instance()->Find(type.domain, type.name, type.version) != nullptr);
}

RetCode CudaEngine::DoOptimize(ir::Graph* graph, utils::SharedResource* resource, RuntimePartitionInfo* info) {
    OptGraph opt_graph(graph, info, resource, &cuda_flags_, &compile_set_);
    auto status = opt_graph.DoOptimize(&device_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph DoOptimeize failed: " << GetRetCodeStr(status);
        return status;
    }

#ifdef PPLNN_ENABLE_CUDA_JIT
    status = CompileCudaModule(graph, resource, info);
#endif

    return RC_SUCCESS;
}

ppl::common::RetCode CudaEngine::CompileCudaModule(ir::Graph* graph, utils::SharedResource* resource,
                                                   RuntimePartitionInfo* info) {
    auto op_compiler_manager = OpCompilerManager::Instance();
    for (auto it = compile_set_.begin(); it != compile_set_.end(); it++) {
        auto node_id = *it;
        ir::Node* op = graph->topo.get()->GetNodeById(node_id);
        auto op_compiler = op_compiler_manager->FindCompiler(op->GetType().name);
        if (op_compiler == nullptr)
            continue;

        const OptKernelOptions options(graph, info, resource, &device_, &cuda_manager_);
        op_compiler->Compile(op, options);
    }

    return RC_SUCCESS;
}

RetCode CudaEngine::ProcessGraph(utils::SharedResource* resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = DoOptimize(graph, resource, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::LoadConstants(*graph, &device_, &info->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "LoadConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

/* -------------------------------------------------------------------------- */

RetCode CudaEngine::SetOutputFormat(CudaEngine* engine, va_list args) {
    auto base = va_arg(args, dataformat_t*);
    auto size = va_arg(args, uint64_t);

    engine->cuda_flags_.output_formats.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
        engine->cuda_flags_.output_formats[i] = base[i];
    }
    return RC_SUCCESS;
}

RetCode CudaEngine::SetOutputType(CudaEngine* engine, va_list args) {
    auto base = va_arg(args, datatype_t*);
    auto size = va_arg(args, uint64_t);

    engine->cuda_flags_.output_types.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
        engine->cuda_flags_.output_types[i] = base[i];
    }
    return RC_SUCCESS;
}

RetCode CudaEngine::SetInputDims(CudaEngine* engine, va_list args) {
    auto& input_dims = engine->cuda_flags_.input_dims;
    auto dims_vec = va_arg(args, utils::Array<int64_t>*);
    auto input_count = va_arg(args, uint64_t);

    input_dims.resize(input_count);
    for (uint64_t i = 0; i < input_count; ++i) {
        const utils::Array<int64_t>& src = dims_vec[i];
        vector<int64_t>& dst = input_dims[i];
        dst.resize(src.size);
        for (uint64_t j = 0; j < src.size; ++j) {
            dst[j] = src.base[j];
        }
    }

    return RC_SUCCESS;
}

RetCode CudaEngine::SetUseDefaultAlgorithms(CudaEngine* engine, va_list args) {
    auto flag = va_arg(args, uint32_t);
    engine->cuda_flags_.quick_select = (flag > 0);
    return RC_SUCCESS;
}

static RetCode ReadFileContent(const char* fname, string* buf) {
    ifstream ifile;

    ifile.open(fname, ios_base::in);
    if (!ifile.is_open()) {
        LOG(ERROR) << "open file[" << fname << "] failed.";
        return RC_NOT_FOUND;
    }

    stringstream ss;
    ss << ifile.rdbuf();
    *buf = ss.str();

    ifile.close();
    return RC_SUCCESS;
}

RetCode CudaEngine::SetQuantization(CudaEngine* engine, va_list args) {
    const char* json_file = va_arg(args, const char*);
    if (!json_file) {
        LOG(ERROR) << "empty quantization info filename.";
        return RC_INVALID_VALUE;
    }

    string json_buffer;
    auto status = ReadFileContent(json_file, &json_buffer);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "read quant info from file[" << json_file << "] failed.";
        return RC_INVALID_VALUE;
    }
    if (json_buffer.empty()) {
        LOG(WARNING) << "empty quant info file[" << json_file << "]. do nothing.";
        return RC_SUCCESS;
    }

    QuantParamParser parser;
    status = parser.ParseBuffer(json_buffer.c_str(), &engine->cuda_flags_.quant_info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse quantization buffer failed: " << GetRetCodeStr(status);
        return status;
    }

    LOG(DEBUG) << "Quant tensor size: " << engine->cuda_flags_.quant_info.tensor_params.size();
    LOG(DEBUG) << "Quant node size: " << engine->cuda_flags_.quant_info.node_params.size();
    return RC_SUCCESS;
}

RetCode CudaEngine::ExportAlgorithms(CudaEngine* engine, va_list args) {
    auto file_path = va_arg(args, const char*);
    engine->cuda_flags_.save_algo_path = std::string(file_path);
    return RC_SUCCESS;
}

RetCode CudaEngine::ImportAlgorithms(CudaEngine* engine, va_list args) {
    auto json_file = va_arg(args, const char*);
    if (!json_file) {
        LOG(WARNING) << "empty algorithm info filename. do nothing.";
        return RC_SUCCESS;
    }

    string json_buffer;
    auto status = ReadFileContent(json_file, &json_buffer);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "read algo info from file[" << json_file << "] failed.";
        return RC_INVALID_VALUE;
    }
    if (json_buffer.empty()) {
        LOG(WARNING) << "empty quant info file[" << json_file << "]. do nothing.";
        return RC_SUCCESS;
    }

    rapidjson::Document d;
    d.Parse(json_buffer.c_str());
    if (d.HasParseError()) {
        LOG(ERROR) << "parse quant file failed: position[" << d.GetErrorOffset() << "], code[" << d.GetParseError()
                   << "]";
        return RC_INVALID_VALUE;
    }

    if (!d.IsObject()) {
        LOG(ERROR) << "quant file content is not an object.";
        return RC_INVALID_VALUE;
    }

    for (auto it = d.MemberBegin(); it != d.MemberEnd(); ++it) {
        const string shape_name(it->name.GetString(), it->name.GetStringLength());
        if (!it->value.IsObject()) {
            LOG(ERROR) << "value of object[" << shape_name << "] is not an object.";
            return RC_INVALID_VALUE;
        }

        CudaArgs::AlgoSelects algo_info;
        for (auto iter = it->value.MemberBegin(); iter != it->value.MemberEnd(); ++iter) {
            const string str_name(iter->name.GetString(), iter->name.GetStringLength());
            if (str_name == "kid") {
                algo_info.kid = iter->value.GetInt();
            } else if (str_name == "splitk") {
                algo_info.splitk = iter->value.GetInt();
            } else if (str_name == "splitf") {
                algo_info.splitf = iter->value.GetInt();
            } else if (str_name == "kname") {
                algo_info.kname.assign(iter->value.GetString(), iter->value.GetStringLength());
            } else {
                LOG(ERROR) << "name of object[" << str_name << "] is not meaningful.";
                return RC_INVALID_VALUE;
            }
        }
        engine->cuda_flags_.alog_selects.insert(make_pair(shape_name, algo_info));
    }
    LOG(DEBUG) << "Algo info size is " << engine->cuda_flags_.alog_selects.size();
    return RC_SUCCESS;
}

CudaEngine::ConfHandlerFunc CudaEngine::conf_handlers_[] = {
    CudaEngine::SetOutputFormat, // CUDA_CONF_SET_OUTPUT_DATA_FORMAT
    CudaEngine::SetOutputType, // CUDA_CONF_SET_OUTPUT_TYPE
    CudaEngine::SetInputDims, // CUDA_CONF_SET_INPUT_DIMS
    CudaEngine::SetUseDefaultAlgorithms, // CUDA_CONF_USE_DEFAULT_ALGORITHMS
    CudaEngine::SetQuantization, // CUDA_CONF_SET_QUANTIZATION
    CudaEngine::ExportAlgorithms, // CUDA_CONF_EXPORT_ALGORITHMS
    CudaEngine::ImportAlgorithms, // CUDA_CONF_IMPORT_ALGORITHMS
};

RetCode CudaEngine::Configure(uint32_t option, ...) {
    if (option >= CUDA_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << CUDA_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::cuda
