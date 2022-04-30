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

#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_IF_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_IF_OP_H_

#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/runtime/runtime_init_info.h"
#include "ppl/nn/params/onnx/if_param.h"
#include "ppl/nn/engines/engine_impl.h"

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace onnx {

class IfOp final {
public:
    IfOp(const ir::Node* node) : node_(node) {}
    ~IfOp();
    ppl::common::RetCode Init(const utils::SharedResource&, ppl::nn::onnx::IfParam*);
    KernelImpl* CreateKernelImpl() const;

private:
    const ir::Node* node_;

    std::shared_ptr<ir::GraphTopo> then_topo_;
    RuntimeGraphInfo then_info_;
    RuntimeAuxInfo then_aux_info_;
    RuntimeInitInfo then_init_info_;
    std::vector<uint32_t> extra_inputs_of_then_graph_; // indices in ir::Node::GetExtraInput()
    std::vector<std::unique_ptr<EngineImpl>> then_engines_;

    std::shared_ptr<ir::GraphTopo> else_topo_;
    RuntimeGraphInfo else_info_;
    RuntimeAuxInfo else_aux_info_;
    RuntimeInitInfo else_init_info_;
    std::vector<uint32_t> extra_inputs_of_else_graph_; // indices in ir::Node::GetExtraInput()
    std::vector<std::unique_ptr<EngineImpl>> else_engines_;
};

}}} // namespace ppl::nn::onnx

#endif
