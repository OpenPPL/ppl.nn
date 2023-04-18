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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_H_

#include <map>
#include <set>

#include "ppl/common/types.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/engines/cuda/engine_options.h"
#include "ppl/nn/engines/cuda/options.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"
#include "ppl/nn/engines/cuda/engine_context.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/quantization/quant_param_parser.h"
#include "ppl/nn/models/pmx/serialization_context.h"

using namespace std;

namespace ppl { namespace nn { namespace cuda {

typedef std::set<nodeid_t> CompileInfo;

struct CudaArgs {
    struct AlgoSelects {
        std::string kname;
        int kid = 0;
        int splitk = 1;
        int splitf = 1;
    };

    bool quick_select = false;
    ppl::common::datatype_t default_kernel_type = 0;
    std::map<std::string, ppl::common::datatype_t> node_types;
    std::vector<std::vector<int64_t>> input_dims;
    std::vector<CudaTensorQuant> tensor_quants;
    std::map<std::string, AlgoSelects> alog_selects;
    QuantParamInfo quant_info;
    const std::vector<int64_t> default_dims{1, 3, 224, 224};
};

struct RefitArgs {
    // std::map<ppl::nn::edgeid_t, void*> edge2val;
    std::map<std::string, edgeid_t> name2edgeid;
    std::map<ppl::nn::edgeid_t, ppl::nn::BufferDesc> edge2buffer;
    std::map<ppl::nn::edgeid_t, ppl::nn::TensorShape> edge2shape;
    std::map<ppl::nn::edgeid_t, std::string> edge2node;
};

class CudaEngine final : public EngineImpl {
public:
    CudaEngine();
    ~CudaEngine();
    ppl::common::RetCode Init(const EngineOptions& options);
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext() override;
    bool Supports(const ir::Node*) const override;
    ppl::common::RetCode ProcessGraph(const utils::SharedResource&, ir::Graph*, RuntimePartitionInfo*) override;
    EngineImpl* Create() override;
#ifdef PPLNN_ENABLE_CUDA_JIT
    ppl::common::RetCode CompileCudaModule(const utils::SharedResource&, const CompileInfo&, ir::Graph*,
                                           RuntimePartitionInfo*);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LoadConstants(const ConstantVisitor&, std::map<edgeid_t, BufferInfo>*) override;
    OptKernel* CreateOptKernel(const ir::Node*) const override;
    ppl::common::RetCode SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream*) const override;
    ppl::common::RetCode DeserializeData(const void*, uint64_t) override;
#endif

private:
    ppl::common::RetCode DoOptimize(const utils::SharedResource&, ir::Graph*, RuntimePartitionInfo*);
    // refit constant tensors
    ppl::common::RetCode FillRefitArgs(RuntimePartitionInfo* info);
    ppl::common::RetCode RefitWeightsImpl(map<edgeid_t, void*>* edge2val);
    ppl::common::RetCode ConvertTorchNameToEdge(const map<string, string>* torch2onnx, const map<string, void*>* name2val, map<edgeid_t, void*>* edge2val);

private:
    /*
      some of them may visit class members.
      defined as member functions can avoid exporting unnecessary APIs
    */
    static ppl::common::RetCode SetKernelType(CudaEngine*, va_list);
    static ppl::common::RetCode SetInputDims(CudaEngine*, va_list);
    static ppl::common::RetCode SetUseDefaultAlgorithms(CudaEngine*, va_list);
    static ppl::common::RetCode SetQuantInfo(CudaEngine*, va_list);
    static ppl::common::RetCode SetExportAlgorithmsHandler(CudaEngine*, va_list);
    static ppl::common::RetCode ImportAlgorithmsFromBuffer(CudaEngine*, va_list);
    static ppl::common::RetCode RefitConstantWeights(CudaEngine*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(CudaEngine*, va_list);
    static ConfHandlerFunc conf_handlers_[ENGINE_CONF_MAX];

private:
    CudaArgs cuda_flags_;
    EngineOptions options_;
    // TODO(WJF): if plain cuda device is used, cuda-memcheck would report illegal errors, may bugs within kernels
    BufferedCudaDevice device_;
    CUDAModuleManager cuda_manager_;
    // update nodes' weights
    RefitArgs refit_args_;
    void* export_algo_arg_ = nullptr;
    void (*export_algo_func_)(const char*, uint64_t, void*) = nullptr;
#ifdef PPLNN_ENABLE_PMX_MODEL
    std::vector<BufferDesc> constant_buffer_blocks_;
#endif
};

}}} // namespace ppl::nn::cuda

#endif
