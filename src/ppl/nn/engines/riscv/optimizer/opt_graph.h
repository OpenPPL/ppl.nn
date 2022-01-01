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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPT_GRAPH_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPT_GRAPH_H_

#include <memory>

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/riscv/riscv_device.h"
#include "ppl/nn/engines/riscv/riscv_engine_options.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace riscv {

class OptGraph final {
public:
    ppl::common::RetCode Init(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*, RiscvEngineOptions* options);
    ppl::common::RetCode DoOptimize(RiscvDevice*);

private:
    ppl::common::RetCode InitKernels(const ir::Graph* graph);
    ppl::common::RetCode InitTensorImpls();
    ppl::common::RetCode TryToInferType(RiscvDevice* device);
    ppl::common::RetCode TryToInferDims(RiscvDevice* device);

private:
    utils::SharedResource* resource_ = nullptr;
    ir::Graph* graph_ = nullptr;
    nn::RuntimePartitionInfo* info_ = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>> tensor_impls_;
    RiscvEngineOptions* options_ = nullptr;
};

}}} // namespace ppl::nn::riscv

#endif
