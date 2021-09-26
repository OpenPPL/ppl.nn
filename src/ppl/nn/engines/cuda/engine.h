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

#include "ppl/common/types.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/engines/cuda/cuda_engine_options.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/quantization/quant_param_parser.h"

using namespace std;

namespace ppl { namespace nn { namespace cuda {

struct CudaArgs {
    CudaArgs() {
        std::vector<uint32_t> default_dims{1, 3, 224, 224};
        input_dims.emplace("", default_dims);
    }

    struct AlgoInfo {
        int kid = 0;
        int splitk = 1;
        int splitf = 1;
    };

    bool quick_select = false;
    ppl::common::datatype_t kernel_default_type = 0;
    std::map<std::string, ppl::common::dataformat_t> output_formats;
    std::map<std::string, ppl::common::datatype_t> output_types;
    std::map<std::string, ppl::common::datatype_t> node_types;
    std::map<std::string, std::vector<uint32_t>> input_dims;
    std::map<std::string, std::vector<CudaTensorQuant>> tensor_quants;
    std::map<std::string, AlgoInfo> alog_selects;
    QuantParamInfo quant_info;
};

class CudaEngine final : public EngineImpl {
public:
    CudaEngine() : EngineImpl("cuda") {}
    ppl::common::RetCode Init(const CudaEngineOptions& options);
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext(const std::string& graph_name) override;
    bool CanRunOp(const ir::Node*) const override;
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;

private:
    ppl::common::RetCode DoOptimize(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);

private:
    /*
      some of them may visit class members.
      defined as member functions can avoid exporting unnecessary APIs
     */
    static ppl::common::RetCode SetOutputFormat(CudaEngine*, va_list);
    static ppl::common::RetCode SetOutputType(CudaEngine*, va_list);
    static ppl::common::RetCode SetCompilerInputDims(CudaEngine*, va_list);
    static ppl::common::RetCode SetUseDefaultAlgorithms(CudaEngine*, va_list);
    static ppl::common::RetCode SetQuantization(CudaEngine*, va_list);
    static ppl::common::RetCode SetAlgorithm(CudaEngine*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(CudaEngine*, va_list);
    static ConfHandlerFunc conf_handlers_[CUDA_CONF_MAX];

private:
    BufferedCudaDevice device_;
    CudaArgs cuda_flags_;
    CudaEngineOptions options_;
};

}}} // namespace ppl::nn::cuda

#endif
