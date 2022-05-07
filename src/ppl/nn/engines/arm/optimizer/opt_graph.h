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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_GRAPH_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_GRAPH_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/arm/arm_device.h"
#include "ppl/nn/engines/arm/engine_options.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"
#include <memory>

namespace ppl { namespace nn { namespace arm {

class OptGraph final {
public:
    ppl::common::RetCode Init(ir::Graph*, RuntimePartitionInfo*, EngineOptions*);
    ppl::common::RetCode DoOptimize(const utils::SharedResource&, ArmDevice*);

private:
    ppl::common::RetCode InitKernels(const ir::Graph* graph);
    ppl::common::RetCode InitTensorImpls();
    ppl::common::RetCode AddReorderOp(const OptKernelOptions& options, const edgeid_t& edge_id, const nodeid_t& node_id,
                                      const int32_t& reorder_type, const ppl::common::dataformat_t& reorder_in_format,
                                      const ppl::common::dataformat_t& reorder_out_format,
                                      const ppl::common::datatype_t& reorder_in_type,
                                      const ppl::common::datatype_t& reorder_out_type);
    ppl::common::RetCode StitchGraph(const OptKernelOptions& options);
    ppl::common::RetCode TryToInferType(ArmDevice* device);
    ppl::common::RetCode TryToInferDims(ArmDevice* device);

    ppl::common::RetCode CreateArmOptKernel(const OptKernelOptions& options, const ir::Node* node,
                                            ArmOptKernel** kernel);

private:
    ir::Graph* graph_ = nullptr;
    RuntimePartitionInfo* info_ = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>> tensor_impls_;
    EngineOptions* options_;
    std::function<EdgeObject*(edgeid_t, uint32_t)> acquire_tensor_func_;
};

}}} // namespace ppl::nn::arm

#endif
