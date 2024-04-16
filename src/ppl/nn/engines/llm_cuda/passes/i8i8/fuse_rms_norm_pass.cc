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

#include "fuse_rms_norm_pass.h"

#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/online_quantize_rms_norm_op.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace i8i8 {

OptPassStatus FuseRMSNormPass(const OptKernelOptions& options)
{
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    for (auto it = options.graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto norm_node = it->Get();
        if (norm_node->GetType().domain == "pmx" && norm_node->GetType().name == "RMSNorm") {

            auto topo = options.graph->topo.get();
            auto kernels = &options.partition_info->kernels;

            auto output_edge = topo->GetEdge(norm_node->GetOutput(0));
            auto q_node = topo->GetNode(output_edge->CreateConsumerIter().Get());

            if (output_edge->CalcConsumerCount() == 1
                && q_node
                && q_node->GetType().domain == "pmx.i8i8"
                && q_node->GetType().name == "OnlineQuantize")
            {
                status.graph_modified = status.graph_modified || true;

                auto q_output_edge = topo->GetEdge(q_node->GetOutput(0));
                auto scale_edge = topo->GetEdge(q_node->GetOutput(1));

                // form: input_edge -> norm_node -> output_edge -> q_node -> (q_output_edge, scale_edge)
                // to  : input_edge -> norm_node -> (q_output_edge, scale_edge)
                norm_node->ReplaceOutput(output_edge->GetId(), q_output_edge->GetId());
                norm_node->InsertOutput(1, scale_edge->GetId());

                kernels->erase(q_node->GetId());
                topo->DelEdge(output_edge->GetId());
                topo->DelNode(q_node->GetId());

                q_output_edge->SetProducer(norm_node->GetId());
                scale_edge->SetProducer(norm_node->GetId());

                norm_node->SetType({"pmx.i8i8", "OnlineQuantizeRMSNorm", 1});
                auto q_norm_kernel = new pmx::I8I8OnlineQuantizeRMSNormOp(norm_node);
                status.retcode = q_norm_kernel->Init(options);
                if (ppl::common::RC_SUCCESS != status.retcode) {
                    LOG(ERROR) << "init kernel[" << q_norm_kernel->GetNode()->GetName()
                        << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
                    return status;
                }
                options.partition_info->kernels[norm_node->GetId()].reset(q_norm_kernel);
                LOG(DEBUG) << "process fuse for RMSNorm[" << norm_node->GetName() << "] success";
            }
        }
    }

    return status;
}

}}}}}
