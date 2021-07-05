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

#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_IF_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_IF_KERNEL_H_

#include "ppl/nn/engines/common/common_kernel_impl.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/runtime/runtime_graph_info.h"

namespace ppl { namespace nn { namespace common {

class IfKernel final : public CommonKernelImpl {
public:
    IfKernel(const ir::Node* node) : CommonKernelImpl(node) {}
    ppl::common::RetCode SetExecutionInfo(const std::shared_ptr<ir::GraphTopo>& then_topo,
                                          const RuntimeGraphInfo* then_info, const RuntimeAuxInfo* then_aux_info,
                                          const std::vector<uint32_t>* extra_inputs_of_then_branch,
                                          const std::shared_ptr<ir::GraphTopo>& else_topo,
                                          const RuntimeGraphInfo* else_info, const RuntimeAuxInfo* else_aux_info,
                                          const std::vector<uint32_t>* extra_inputs_of_else_branch,
                                          utils::SharedResource*);

protected:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    RuntimeImpl then_branch_, else_branch_;
    const std::vector<uint32_t>* extra_inputs_of_then_branch_ = nullptr;
    const std::vector<uint32_t>* extra_inputs_of_else_branch_ = nullptr;
};

}}} // namespace ppl::nn::common

#endif
