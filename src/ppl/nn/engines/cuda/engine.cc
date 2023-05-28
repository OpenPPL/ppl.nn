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
#include <fstream>

#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/engines/cuda/optimizer/opt_graph.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/module/op_compile_manager.h"
#include "ppl/nn/quantization/quant_param_parser.h"
#include "ppl/nn/utils/array.h"
#include "ppl/nn/common/logger.h"
// refit weights
#include "cudakernel/nn/conv/group_padding.h"
#include "ppl/common/destructor.h"

#include "rapidjson/document.h"
#include "rapidjson/error/error.h"
#include "rapidjson/error/en.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/cuda/macros.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_engine_generated.h"
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
    return device_.Init(options.device_id, options.mm_policy);
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

#ifdef PPLNN_ENABLE_CUDA_JIT
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
#endif

static void ExportAlgorithmsInfo(const map<string, CudaArgs::AlgoSelects>& algos,
                                 void (*func)(const char*, uint64_t, void*), void* arg) {
    rapidjson::Document d;
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();

    d.SetObject();

    for (auto s = algos.begin(); s != algos.end(); ++s) {
        auto& item = s->second;
        rapidjson::Value object(rapidjson::kObjectType);
        object.AddMember("kname", rapidjson::StringRef(item.kname.data(), item.kname.size()), allocator);
        object.AddMember("kid", item.kid, allocator);
        object.AddMember("splitk", item.splitk, allocator);
        object.AddMember("splitf", item.splitf, allocator);
        rapidjson::Value key_info(s->first.data(), s->first.size(), allocator);
        d.AddMember(key_info, object, allocator);
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    func(buffer.GetString(), buffer.GetSize(), arg);
}

RetCode CudaEngine::FillRefitArgs(RuntimePartitionInfo* info) {
    for (auto iter = info->name2edgeid.begin(); iter != info->name2edgeid.end(); ++iter) {
        refit_args_.name2edgeid.insert(make_pair(iter->first, iter->second));
    }
    for (auto iter = info->edge2node.begin(); iter != info->edge2node.end(); ++iter) {
        refit_args_.edge2node.insert(make_pair(iter->first, iter->second));
    }

    for (auto c = info->constants.begin(); c != info->constants.end(); ++c) {
        auto& src = c->second;
        refit_args_.edge2shape.insert(make_pair(c->first, *src.GetShape()));
        auto ret_pair = refit_args_.edge2buffer.insert(make_pair(c->first, src.GetBufferDesc()));
        if (!ret_pair.second) {
            return RC_INVALID_VALUE;
        }
    }
    return RC_SUCCESS;
}

RetCode CudaEngine::ProcessGraph(const utils::SharedResource& resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = DoOptimize(resource, graph, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    if (export_algo_func_) {
        ExportAlgorithmsInfo(cuda_flags_.alog_selects, export_algo_func_, export_algo_arg_);
    }
    
    // fill refit args, only constant tensors will be recorded
    FillRefitArgs(info);

    return RC_SUCCESS;
}

ppl::common::RetCode CudaEngine::RefitWeightsImpl(map<edgeid_t, const void*>* edge2val) {
    // only one partition is used
    auto dev = &device_;
    for (auto iter = edge2val->begin(); iter != edge2val->end(); ++iter) {
        auto edge_id = iter->first;
        auto data_ptr = iter->second;
        auto shape_ref = refit_args_.edge2shape.find(edge_id);
        if (shape_ref == refit_args_.edge2shape.end()) { return RC_INVALID_VALUE; }
        auto dst_shape = shape_ref->second;
        auto src_shape(dst_shape);
        src_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        src_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        auto& buf_info = refit_args_.edge2buffer[edge_id];
        if (refit_args_.edge2node[edge_id] == "Conv") { // conv
            if (cuda_flags_.default_kernel_type == ppl::common::DATATYPE_FLOAT16) { // only support fp16 conv now
                // allocate temp buffer to finish convert from host
                BufferDesc tmp_buffer_desc;
                auto status = dev->Realloc(dst_shape, &tmp_buffer_desc);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
                    return status;
                }
                Destructor __tmp_buffer_guard__([dev, &tmp_buffer_desc]() -> void {
                    dev->Free(&tmp_buffer_desc);
                });
                dev->GetDataConverter()->ConvertFromHost(&tmp_buffer_desc, dst_shape, data_ptr, src_shape);
                // conv related convert
                conv_param_t tmp_conv_param;
                tmp_conv_param.num_flt = dst_shape.GetDim(0);
                tmp_conv_param.num_chl = dst_shape.GetDim(1);
                tmp_conv_param.flt_height = dst_shape.GetDim(2);
                tmp_conv_param.flt_width = dst_shape.GetDim(3);
                tmp_conv_param.num_grp = 1; // TODO(WJF): record related nodes' parameters
                PPLCUDAConvolutionCvtFlt(dev->GetStream(), buf_info.addr, tmp_buffer_desc.addr,
                        ppl::common::DATATYPE_FLOAT16, tmp_conv_param); 
            } else {
                return RC_UNSUPPORTED;
            }
        } else { // matmul just convert and copy
            dev->GetDataConverter()->ConvertFromHost(&buf_info, dst_shape, data_ptr, src_shape);
        }
    }
    dev->Sync();
    // cudaDeviceSynchronize();
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
        return RC_SUCCESS;
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
        auto status = dev->CopyFromHost(&buf, data, size);
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

    auto param = opt_kernel->GetCommparam();
    auto& quant = cuda_flags_.tensor_quants;
    param->cuda_tensor_info = const_cast<std::vector<ppl::nn::cuda::CudaTensorQuant>*>(&quant);

    return opt_kernel;
}

ppl::common::RetCode CudaEngine::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    auto& quant = cuda_flags_.tensor_quants;
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<pmx::cuda::CudaTensorQuant>> quant_vec;
    quant_vec.resize(ctx.seq2eid.size());

    for (uint32_t i = 0; i < ctx.seq2eid.size(); ++i) {
        auto original_eid = ctx.seq2eid[i];
        auto fb_tensor_quant = pmx::cuda::CreateCudaTensorQuantDirect(
            builder,
            quant[original_eid].format,
            quant[original_eid].type,
            quant[original_eid].per_channel,
            quant[original_eid].bit_width,
            &quant[original_eid].scale,
            &quant[original_eid].zero_point);
        quant_vec[i] = fb_tensor_quant;
    }
    auto fb_tensor_quants = pmx::cuda::CreateTensorQuantsDirect(builder, &quant_vec);
    auto fb_cuda_engine_param = pmx::cuda::CreateCudaEngineParam(builder, pmx::cuda::CudaEngineParamType_TensorQuants, fb_tensor_quants.Union());
    pmx::cuda::FinishCudaEngineParamBuffer(builder, fb_cuda_engine_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode CudaEngine::DeserializeData(const void* base, uint64_t size) {
    auto fb_cuda_engine_param = pmx::cuda::GetCudaEngineParam(base);
    auto fb_tensor_quants = fb_cuda_engine_param->value_as_TensorQuants()->tensor_quants();
    for (uint32_t i = 0; i < fb_tensor_quants->size(); ++i) {
        CudaTensorQuant quant;
        quant.format = fb_tensor_quants->Get(i)->format();
        quant.type = fb_tensor_quants->Get(i)->type();
        quant.per_channel = fb_tensor_quants->Get(i)->per_channel();
        quant.bit_width = fb_tensor_quants->Get(i)->bit_width();
        ppl::nn::pmx::utils::Fbvec2Stdvec(fb_tensor_quants->Get(i)->scale(), &quant.scale);
        ppl::nn::pmx::utils::Fbvec2Stdvec(fb_tensor_quants->Get(i)->zero_point(), &quant.scale);
        cuda_flags_.tensor_quants.push_back(quant);
    }
    return ppl::common::RC_SUCCESS;
}

#endif

/* -------------------------------------------------------------------------- */

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

RetCode CudaEngine::SetExportAlgorithmsHandler(CudaEngine* engine, va_list args) {
    typedef void (*callback_func_t)(const char*, uint64_t, void*);
    engine->export_algo_func_ = va_arg(args, callback_func_t);
    engine->export_algo_arg_ = va_arg(args, void*);
    return RC_SUCCESS;
}

static RetCode ImportAlgorithmsImpl(const char* json_buffer, uint64_t buffer_size,
                                    map<string, CudaArgs::AlgoSelects>* algo_selected) {
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
        auto ref = it->value.FindMember("kid");
        if (ref != it->value.MemberEnd()) {
            algo_info.kid = ref->value.GetInt();
        }
        ref = it->value.FindMember("splitk");
        if (ref != it->value.MemberEnd()) {
            algo_info.splitk = ref->value.GetInt();
        }
        ref = it->value.FindMember("splitf");
        if (ref != it->value.MemberEnd()) {
            algo_info.splitf = ref->value.GetInt();
        }
        ref = it->value.FindMember("kname");
        if (ref != it->value.MemberEnd()) {
            algo_info.kname.assign(ref->value.GetString(), ref->value.GetStringLength());
        }
        algo_selected->insert(make_pair(shape_name, algo_info));
    }

    LOG(DEBUG) << "Algo info size is " << algo_selected->size();
    return RC_SUCCESS;
}

ppl::common::RetCode CudaEngine::ImportAlgorithmsFromBuffer(CudaEngine* engine, va_list args) {
    auto json_buffer = va_arg(args, const char*);
    auto buffer_size = va_arg(args, uint64_t);
    return ImportAlgorithmsImpl(json_buffer, buffer_size, &engine->cuda_flags_.alog_selects);
}

ppl::common::RetCode CudaEngine::ConvertTorchNameToEdge(const map<string, string>* torch2onnx,
    const map<string, const void*>* name2val, map<edgeid_t, const void*>* edge2val) {
    for (auto iter = name2val->begin(); iter != name2val->end(); ++iter) {
        auto torch_name = iter->first;
        auto onnx_ref = torch2onnx->find(torch_name);
        if (onnx_ref == torch2onnx->end()) {
            LOG(ERROR) << "Missing torch name --> onnx name mapping!";
            return RC_INVALID_VALUE;
        }
        auto onnx_name = onnx_ref->second;
        auto edgeid_ref = refit_args_.name2edgeid.find(onnx_name);
        if (edgeid_ref == refit_args_.name2edgeid.end()) {
            LOG(ERROR) << "Unkown onnx tensor name!";
            return RC_INVALID_VALUE;
        } else {
            auto edge_id = edgeid_ref->second;
            auto data_ptr = iter->second;
            auto ret_pair = edge2val->insert(make_pair(edge_id, data_ptr));
            if (!ret_pair.second) {
                LOG(ERROR) << "Record edge_id --> data_ptr failed!";
                return RC_INVALID_VALUE;
            }
        }
    }
    return RC_SUCCESS;
}

typedef std::map<std::string, std::string> MapOfString;
typedef std::map<std::string, const void*> MapOfPointer;
ppl::common::RetCode CudaEngine::RefitConstantWeights(CudaEngine* engine, va_list args) {
    auto torch2onnx = va_arg(args, MapOfString*);
    auto name2val = va_arg(args, MapOfPointer*);
    map<edgeid_t, const void*> edge2val;
    auto status = engine->ConvertTorchNameToEdge(torch2onnx, name2val, &edge2val);
    if (status != RC_SUCCESS) return RC_UNSUPPORTED;
    status = engine->RefitWeightsImpl(&edge2val);
    return status;
}

CudaEngine::ConfHandlerFunc CudaEngine::conf_handlers_[] = {
    CudaEngine::SetKernelType, // ENGINE_CONF_USE_DEFAULT_KERNEL_TYPE
    CudaEngine::SetInputDims, // ENGINE_CONF_SET_INPUT_DIMS
    CudaEngine::SetUseDefaultAlgorithms, // ENGINE_CONF_USE_DEFAULT_ALGORITHMS
    CudaEngine::SetQuantInfo, // ENGINE_CONF_SET_QUANT_INFO
    CudaEngine::SetExportAlgorithmsHandler, // ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER
    CudaEngine::ImportAlgorithmsFromBuffer, // ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER
    CudaEngine::RefitConstantWeights, // ENGINE_CONF_REFIT_CONSTANT_WEIGHTS
};

RetCode CudaEngine::Configure(uint32_t option, ...) {
    if (option >= ENGINE_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << (uint32_t)ENGINE_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::cuda
