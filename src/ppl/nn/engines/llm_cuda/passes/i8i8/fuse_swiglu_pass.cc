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

#include "fuse_swiglu_pass.h"

#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_dequantize_swiglu_quantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_dequantize_op.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace i8i8 {

OptPassStatus FuseSwiGLUPass(const OptKernelOptions& options)
{
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    for (auto it = options.graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto swiglu_node = it->Get();
        if (swiglu_node->GetType().domain == "opmx" && swiglu_node->GetType().name == "SwiGLU") {

            auto topo = options.graph->topo.get();
            auto kernels = &options.partition_info->kernels;

            auto output_edge = topo->GetEdge(swiglu_node->GetOutput(0));
            auto input_edge = topo->GetEdge(swiglu_node->GetInput(0));

            auto q_node = topo->GetNode(output_edge->CreateConsumerIter().Get());
            auto dq_node = topo->GetNode(input_edge->GetProducer());

            if (output_edge->CalcConsumerCount() == 1
                && q_node
                && q_node->GetType().domain == "opmx.i8i8"
                && q_node->GetType().name == "OnlineQuantize"
                && input_edge->CalcConsumerCount() == 1
                && dq_node
                && dq_node->GetType().domain == "opmx.i8i8"
                && dq_node->GetType().name == "OnlineDequantize"
            )
            {
                auto dq_kernel = (opmx::I8I8OnlineDequantizeOp*)options.partition_info->kernels[dq_node->GetId()].get();
                if (dq_kernel->GetParam()->bias_term)
                    continue;

                status.graph_modified = status.graph_modified || true;

                auto q_output_edge = topo->GetEdge(q_node->GetOutput(0));
                auto scale_edge = topo->GetEdge(q_node->GetOutput(1));

                auto dq_input_edge = topo->GetEdge(dq_node->GetInput(0));
                auto dq_scale_outer_edge = topo->GetEdge(dq_node->GetInput(1));
                auto dq_scale_inner_edge = topo->GetEdge(dq_node->GetInput(2));

                // form: (dq_input_edge, scale_outer, scale_inner) -> dq_node -> input_edge -> swiglu_node -> output_edge -> q_node -> (q_output_edge, scale_edge)
                // to  : (dq_input_edge, scale_outer, scale_inner) -> swiglu_node -> (q_output_edge, scale_edge)
                swiglu_node->ReplaceOutput(output_edge->GetId(), q_output_edge->GetId());
                swiglu_node->AddOutput(scale_edge->GetId());

                swiglu_node->ClearInputs();
                swiglu_node->AddInput(dq_input_edge->GetId());
                swiglu_node->AddInput(dq_scale_outer_edge->GetId());
                swiglu_node->AddInput(dq_scale_inner_edge->GetId());

                dq_input_edge->DelConsumer(dq_node->GetId());
                dq_input_edge->AddConsumer(swiglu_node->GetId());

                dq_scale_outer_edge->DelConsumer(dq_node->GetId());
                dq_scale_outer_edge->AddConsumer(swiglu_node->GetId());

                dq_scale_inner_edge->DelConsumer(dq_node->GetId());
                dq_scale_inner_edge->AddConsumer(swiglu_node->GetId());

                q_output_edge->SetProducer(swiglu_node->GetId());
                scale_edge->SetProducer(swiglu_node->GetId());

                kernels->erase(q_node->GetId());
                topo->DelEdge(output_edge->GetId());
                topo->DelNode(q_node->GetId());

                kernels->erase(dq_node->GetId());
                topo->DelEdge(input_edge->GetId());
                topo->DelNode(dq_node->GetId());

                swiglu_node->SetType({"opmx.i8i8", "OnlineDequantizeSwiGLUQuantize", 1});
                auto dq_swiglu_q_kernel = new opmx::I8I8OnlineDequantizeSwiGLUQuantizeOp(swiglu_node);
                status.retcode = dq_swiglu_q_kernel->Init(options);
                if (ppl::common::RC_SUCCESS != status.retcode) {
                    LOG(ERROR) << "init kernel[" << dq_swiglu_q_kernel->GetNode()->GetName()
                        << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
                    return status;
                }
                options.partition_info->kernels[swiglu_node->GetId()].reset(dq_swiglu_q_kernel);
                LOG(DEBUG) << "process fuse for SwiGLU[" << swiglu_node->GetName() << "] success";
            }
        }
    }

    return status;
}

}}}}}
