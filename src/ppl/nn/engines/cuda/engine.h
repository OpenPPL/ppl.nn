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
#include "ppl/nn/engines/cuda/cuda_engine_options.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"
#include "ppl/nn/engines/cuda/engine_context.h"
#include "ppl/nn/quantization/quant_param_parser.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"

using namespace std;

namespace ppl { namespace nn { namespace cuda {

typedef std::set<nodeid_t> CompileInfo;

struct CudaArgs {
    struct AlgoSelects {
        std::string kname = "";
        int kid = 0;
        int splitk = 1;
        int splitf = 1;
    };

    bool quick_select;
    std::string save_algo_path = "";
    ppl::common::datatype_t kernel_default_type = 0;
    std::map<std::string, ppl::common::datatype_t> node_types;
    std::vector<ppl::common::dataformat_t> output_formats;
    std::vector<ppl::common::datatype_t> output_types;
    std::vector<std::vector<int64_t>> input_dims;
    std::map<std::string, std::vector<CudaTensorQuant>> tensor_quants;
    std::map<std::string, AlgoSelects> alog_selects;
    QuantParamInfo quant_info;
    const std::vector<int64_t> default_dims{1, 3, 224, 224};
};

class CudaEngine final : public EngineImpl {
public:
    CudaEngine() : EngineImpl("cuda") {}
    ppl::common::RetCode Init(const CudaEngineOptions& options);
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext() override;
    bool Supports(const ir::Node*) const override;
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;
    ppl::common::RetCode CompileCudaModule(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);

private:
    ppl::common::RetCode DoOptimize(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);

private:
    /*
      some of them may visit class members.
      defined as member functions can avoid exporting unnecessary APIs
    */
    static ppl::common::RetCode SetOutputFormat(CudaEngine*, va_list);
    static ppl::common::RetCode SetOutputType(CudaEngine*, va_list);
    static ppl::common::RetCode SetInputDims(CudaEngine*, va_list);
    static ppl::common::RetCode SetUseDefaultAlgorithms(CudaEngine*, va_list);
    static ppl::common::RetCode SetQuantization(CudaEngine*, va_list);
    static ppl::common::RetCode ExportAlgorithms(CudaEngine*, va_list);
    static ppl::common::RetCode ImportAlgorithms(CudaEngine*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(CudaEngine*, va_list);
    static ConfHandlerFunc conf_handlers_[CUDA_CONF_MAX];

private:
    BufferedCudaDevice device_;
    CudaArgs cuda_flags_;
    CudaEngineOptions options_;
    CUDAModuleManager cuda_manager_;
    CompileInfo compile_set_;
};

}}} // namespace ppl::nn::cuda

#endif
