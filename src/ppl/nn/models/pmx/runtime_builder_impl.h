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

#ifndef _ST_HPC_PPL_NN_MODELS_PMX_RUNTIME_BUILDER_IMPL_H_
#define _ST_HPC_PPL_NN_MODELS_PMX_RUNTIME_BUILDER_IMPL_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/utils/shared_resource.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/runtime/partial_runtime_creator.h"
#include "ppl/nn/models/pmx/runtime_builder.h"
#include "ppl/nn/models/pmx/runtime_builder_options.h"

namespace ppl { namespace nn { namespace pmx {

class RuntimeBuilderImpl final : public RuntimeBuilder {
public:
    RuntimeBuilderImpl();
    ~RuntimeBuilderImpl();
    ppl::common::RetCode Init(const char* model_file, Engine** engines, uint32_t engine_num) override;
    ppl::common::RetCode Init(const char* model_buf, uint64_t buf_len, Engine** engines, uint32_t engine_num) override;
    ppl::common::RetCode Configure(uint32_t, ...) override;
    ppl::common::RetCode Preprocess() override;
    Runtime* CreateRuntime() override;
    Runtime* CreateRuntime(const char** begin_ops, uint32_t begin_op_num, const char** end_ops,
                           uint32_t end_op_num) override;
    ppl::common::RetCode Serialize(const char* output_file, const char* fmt) const override;

private:
    static ppl::common::RetCode ReserveTensor(RuntimeBuilderImpl*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(RuntimeBuilderImpl*, va_list);
    static ConfHandlerFunc conf_handlers_[PRB_CONF_MAX];

private:
    utils::SharedResource resource_;
    std::shared_ptr<ir::GraphTopo> topo_;
    std::shared_ptr<RuntimeGraphInfo> graph_info_;
    std::shared_ptr<RuntimeAuxInfo> aux_info_;
    RuntimeInitInfo init_info_;
    PartialRuntimeCreator partial_runtime_creator_;
};

}}} // namespace ppl::nn::pmx

#endif
