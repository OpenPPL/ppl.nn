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
#include <algorithm>

#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/engines/cuda/optimizer/opt_graph.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/module/op_compile_manager.h"
#include "ppl/nn/quantization/quant_param_parser.h"
#include "ppl/nn/utils/array.h"
#include "ppl/nn/utils/utils.h"
#include "ppl/nn/common/logger.h"

#include "rapidjson/document.h"
#include "rapidjson/error/error.h"
#include "rapidjson/error/en.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/cuda/macros.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

CudaEngine::CudaEngine() : EngineImpl("cuda") {
    if (OptKernelCreatorManager::GetInstance()->GetSize() == 0) {
        LOG(WARNING) << "Empty op implementation set. Did you forget to call `ppl::nn::cuda::RegisterBuiltinOpImpls()` "
                        "before creating cuda engines?";
    }
}

CudaEngine::~CudaEngine() {
#ifdef PPLNN_ENABLE_PMX_MODEL
    for (auto b = constant_buffer_blocks_.begin(); b != constant_buffer_blocks_.end(); ++b) {
        device_.Free(&(*b));
    }
#endif
}

RetCode CudaEngine::Init(const EngineOptions& options) {
    options_ = options;
    return device_.Init(options.device_id);
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
    return (OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version) != nullptr);
}

RetCode CudaEngine::DoOptimize(const utils::SharedResource& resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    CompileInfo compile_set;
    OptGraph opt_graph(graph, info, &cuda_flags_, &compile_set);
    auto status = opt_graph.DoOptimize(resource, &device_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph DoOptimeize failed: " << GetRetCodeStr(status);
        return status;
    }

#ifdef PPLNN_ENABLE_CUDA_JIT
    status = CompileCudaModule(resource, compile_set, graph, info);
#endif

    return RC_SUCCESS;
}

ppl::common::RetCode CudaEngine::CompileCudaModule(const utils::SharedResource& resource,
                                                   const CompileInfo& compile_set, ir::Graph* graph,
                                                   RuntimePartitionInfo* info) {
    auto op_compiler_manager = OpCompilerManager::Instance();
    for (auto it = compile_set.begin(); it != compile_set.end(); ++it) {
        auto node_id = *it;
        ir::Node* op = graph->topo.get()->GetNode(node_id);
        if (!op)
            continue;
        auto op_compiler = op_compiler_manager->FindCompiler(op->GetType().name);
        if (op_compiler == nullptr)
            continue;

        const OptKernelOptions options(graph, info, &resource, &device_, &cuda_manager_);
        op_compiler->Compile(op, options);
    }

    return RC_SUCCESS;
}

RetCode CudaEngine::ProcessGraph(const utils::SharedResource& resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = DoOptimize(resource, graph, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

EngineImpl* CudaEngine::Create() {
    return static_cast<EngineImpl*>(EngineFactory::Create(options_));
}

#ifdef PPLNN_ENABLE_PMX_MODEL
static inline uint64_t Align(uint64_t v, uint64_t alignment) {
    return (v + alignment - 1) & (~(alignment - 1));
}

RetCode CudaEngine::LoadConstants(const ConstantVisitor& visitor, map<edgeid_t, BufferInfo>* eid2info) {
    uint64_t total_bytes = 0;
    auto status = visitor.ForEach([&total_bytes](const void*, uint64_t bytes) -> RetCode {
        total_bytes += Align(bytes, CUDA_DEFAULT_ALIGNMENT);
        return true;
    });
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "calc total bytes of constant failed: " << GetRetCodeStr(status);
        return status;
    }

    if (total_bytes == 0) {
        return RC_SUCCESS;
    }

    BufferDesc block;
    status = device_.Realloc(total_bytes, &block);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc [" << total_bytes << "] bytes failed: " << GetRetCodeStr(status);
        return status;
    }

    uint64_t offset = 0;
    status = visitor.ForEach([this, eid2info, &block, &offset](const ir::Edge* edge, const void* data, uint64_t size,
                                                               const TensorShape& shape) -> RetCode {
        auto dev = &device_;

        BufferDesc buf;
        buf.addr = (char*)block.addr + offset;
        auto status = dev->CopyFromHost(&buf, data, shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy data of constant[" << edge->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        BufferInfo info;
        info.SetBuffer(buf, dev);
        auto ret_pair = eid2info->emplace(edge->GetId(), std::move(info));
        if (!ret_pair.second) {
            LOG(ERROR) << "constant[" << edge->GetName() << "] already exists.";
            return RC_EXISTS;
        }

        offset += Align(size, CUDA_DEFAULT_ALIGNMENT);
        return RC_SUCCESS;
    });

    if (status == RC_SUCCESS) {
        constant_buffer_blocks_.push_back(block);
    } else {
        device_.Free(&block);
    }

    return status;
}

OptKernel* CudaEngine::CreateOptKernel(const ir::Node* node) const {
    auto& type = node->GetType();
    auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for node[" << node->GetName() << "] of type[" << type.domain << ":"
                   << type.name << ":" << type.version << "]";
        return nullptr;
    }

    auto opt_kernel = (*creator)(node);
    if (!opt_kernel) {
        LOG(ERROR) << "create kernel[" << node->GetName() << "] failed: oom.";
        return nullptr;
    }

    return opt_kernel;
}
#endif

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

RetCode CudaEngine::SetKernelType(CudaEngine* engine, va_list args) {
    engine->cuda_flags_.default_kernel_type = va_arg(args, datatype_t);
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

RetCode CudaEngine::SetQuantInfo(CudaEngine* engine, va_list args) {
    const char* json_str = va_arg(args, const char*);
    uint64_t json_size = va_arg(args, uint64_t);
    if (!json_str || json_size == 0) {
        LOG(ERROR) << "empty quantization info string.";
        return RC_INVALID_VALUE;
    }

    auto status = QuantParamParser::ParseBuffer(json_str, json_size, &engine->cuda_flags_.quant_info);
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

    utils::Buffer json_buffer;
    auto status = utils::ReadFileContent(json_file, &json_buffer);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "read algo info from file[" << json_file << "] failed.";
        return RC_INVALID_VALUE;
    }
    if (json_buffer.GetSize() == 0) {
        LOG(WARNING) << "empty quant info file[" << json_file << "]. do nothing.";
        return RC_SUCCESS;
    }

    return ImportAlgorithmsImpl(engine, (const char*)(json_buffer.GetData()), json_buffer.GetSize());
}

ppl::common::RetCode CudaEngine::ImportAlgorithmsFromBuffer(CudaEngine* engine, va_list args) {
    auto json_buffer = va_arg(args, const char*);
    auto buffer_size = va_arg(args, size_t);
    return ImportAlgorithmsImpl(engine, json_buffer, buffer_size);
}

RetCode CudaEngine::ImportAlgorithmsImpl(CudaEngine* engine, const char* json_buffer, size_t buffer_size) {
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(json_buffer, buffer_size);
    if (!ok) {
        LOG(ERROR) << "parse quant buffer failed: [" << rapidjson::GetParseError_En(ok.Code()) << "], offset["
                   << ok.Offset() << "]";
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
    CudaEngine::SetOutputFormat, // ENGINE_CONF_SET_OUTPUT_DATA_FORMAT
    CudaEngine::SetOutputType, // ENGINE_CONF_SET_OUTPUT_TYPE
    CudaEngine::SetKernelType, // ENGINE_CONF_USE_DEFAULT_KERNEL_TYPE
    CudaEngine::SetInputDims, // ENGINE_CONF_SET_INPUT_DIMS
    CudaEngine::SetUseDefaultAlgorithms, // ENGINE_CONF_USE_DEFAULT_ALGORITHMS
    CudaEngine::SetQuantInfo, // ENGINE_CONF_SET_QUANT_INFO
    CudaEngine::ExportAlgorithms, // ENGINE_CONF_EXPORT_ALGORITHMS
    CudaEngine::ImportAlgorithms, // ENGINE_CONF_IMPORT_ALGORITHMS
    CudaEngine::ImportAlgorithmsFromBuffer, // ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER
};

RetCode CudaEngine::Configure(uint32_t option, ...) {
    if (option >= ENGINE_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << ENGINE_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::cuda
