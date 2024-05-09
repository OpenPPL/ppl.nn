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

#include "fuse_split_pass.h"

#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_dequantize_reshape_split_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_dequantize_op.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace i8i8 {

OptPassStatus FuseSplitPass(const OptKernelOptions& options)
{
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    for (auto it = options.graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto split_node = it->Get();
        if (split_node->GetType().domain == "" && split_node->GetType().name == "Split") {

            auto topo = options.graph->topo.get();
            auto kernels = &options.partition_info->kernels;
            auto data = options.graph->data.get();

            auto input_edge = topo->GetEdge(split_node->GetInput(0));
            auto split_edge = topo->GetEdge(split_node->GetInput(1));
            auto reshape_node = topo->GetNode(input_edge->GetProducer());

            // is this a static split?
            std::vector<int64_t> constant_split_data;
            {
                auto split_data_it = data->constants.find(split_edge->GetId());
                const int64_t* split_data = nullptr;
                if (split_data_it != data->constants.end()) {
                    split_data = (const int64_t*)split_data_it->second.data.GetData();
                }

                if (split_data != nullptr) {
                    auto split_shape_it = data->shapes.find(split_edge->GetId());
                    if (split_shape_it != data->shapes.end()) {
                        auto& split_shape = split_shape_it->second;
                        constant_split_data.assign(split_data, split_data + split_shape.dims[0]);
                    }
                }
            }
            if (constant_split_data.size() == 0)
                continue;

            // find it's param
            ppl::nn::onnx::SplitParam* split_param = nullptr;
            {
                auto param_ref = data->attrs.find(split_node->GetId());
                if (param_ref != data->attrs.end()) {
                    split_param = (ppl::nn::onnx::SplitParam*)param_ref->second.get();
                }
            }
            if (split_param == nullptr)
                continue;

            if (!reshape_node)
                continue;
            auto rs_input_edge = topo->GetEdge(reshape_node->GetInput(0));
            auto shape_edge = topo->GetEdge(reshape_node->GetInput(1));
            auto dq_node = topo->GetNode(rs_input_edge->GetProducer());

            if (input_edge->CalcConsumerCount() == 1
                && reshape_node->GetType().domain == ""
                && reshape_node->GetType().name == "Reshape"
                && dq_node
                && dq_node->GetType().domain == "opmx.i8i8"
                && dq_node->GetType().name == "OnlineDequantize")
            {
                // is this a static reshape?
                std::vector<int64_t> constant_shape_data;
                {
                    auto shape_data_it = data->constants.find(shape_edge->GetId());
                    const int64_t* shape_data = nullptr;
                    if (shape_data_it != data->constants.end()) {
                        shape_data = (const int64_t*)shape_data_it->second.data.GetData();
                    }

                    if (shape_data != nullptr) {
                        auto shape_shape_it = data->shapes.find(shape_edge->GetId());
                        if (shape_shape_it != data->shapes.end()) {
                            auto& shape_shape = shape_shape_it->second;
                            constant_shape_data.assign(shape_data, shape_data + shape_shape.dims[0]);
                        }
                    }
                }
                if (constant_shape_data.size() == 0)
                    continue;

                // We only accept reshape like (0,0,...,0,A,B,C,D,E,...)
                //                             axis here -|
                // Where A is -1 or postive
                // And B,C,D,E.. must be postive
                int64_t prefix_dim = 0;
                int64_t suffix_dim = 1;
                int64_t dim_count = (int64_t)constant_shape_data.size();
                int64_t axis = split_param->axis < 0 ? split_param->axis + dim_count : split_param->axis;
                if (constant_shape_data[axis] == 0)
                    continue;
                for (int64_t i = 0; i < axis; ++i)
                    prefix_dim += constant_shape_data[i];
                for (int32_t i = axis + 1; i < dim_count; ++i)
                    suffix_dim *= constant_shape_data[i];
                if (prefix_dim != 0 || suffix_dim <= 0)
                    continue;

                // now we can do optimize
                status.graph_modified = status.graph_modified || true;

                auto dq_kernel = (opmx::I8I8OnlineDequantizeOp*)options.partition_info->kernels[dq_node->GetId()].get();
                auto dq_input_edge = topo->GetEdge(dq_node->GetInput(0));
                auto dq_scale_outer_edge = topo->GetEdge(dq_node->GetInput(1));
                auto dq_scale_inner_edge = topo->GetEdge(dq_node->GetInput(2));
                auto dq_bias_edge = dq_kernel->GetParam()->bias_term ? topo->GetEdge(dq_node->GetInput(3)) : nullptr;

                // form: dq_input_edge       -> dq_node -> rs_input_edge -> reshape_node ->  input_edge -> split_node
                //       dq_scale_outer_edge -|             shape_edge   -/                  split_edge -/
                //       dq_scale_inner_edge -|
                //              dq_bias_edge -/
                // to  : dq_input_edge       -> split_node
                //       dq_scale_outer_edge -|
                //       dq_scale_inner_edge -|
                //              dq_bias_edge -/
                split_node->ReplaceInput(input_edge->GetId(), dq_input_edge->GetId());
                split_node->ReplaceInput(split_edge->GetId(), dq_scale_outer_edge->GetId());
                split_node->AddInput(dq_scale_inner_edge->GetId());
                if (dq_bias_edge)
                    split_node->AddInput(dq_bias_edge->GetId());

                dq_input_edge->DelConsumer(dq_node->GetId());
                dq_input_edge->AddConsumer(split_node->GetId());

                dq_scale_outer_edge->DelConsumer(dq_node->GetId());
                dq_scale_outer_edge->AddConsumer(split_node->GetId());

                dq_scale_inner_edge->DelConsumer(dq_node->GetId());
                dq_scale_inner_edge->AddConsumer(split_node->GetId());

                shape_edge->DelConsumer(reshape_node->GetId()); // do not delete this edge
                
                split_edge->DelConsumer(split_node->GetId()); // do not delete this edge

                if (dq_bias_edge) {
                    dq_bias_edge->DelConsumer(dq_node->GetId());
                    dq_bias_edge->AddConsumer(split_node->GetId());
                }

                kernels->erase(dq_node->GetId());
                kernels->erase(reshape_node->GetId());
                topo->DelEdge(input_edge->GetId());
                topo->DelNode(dq_node->GetId());
                topo->DelEdge(rs_input_edge->GetId());
                topo->DelNode(reshape_node->GetId());

                split_node->SetType({"opmx.i8i8", "OnlineDequantizeReshapeSplit", 1});
                auto dq_split_kernel = new opmx::I8I8OnlineDequantizeReshapeSplitOp(split_node);
                // set params
                dq_split_kernel->GetParam()->bias_term = dq_bias_edge != nullptr;
                dq_split_kernel->GetParam()->shape.assign(constant_shape_data.begin(), constant_shape_data.end());
                dq_split_kernel->GetParam()->split.assign(constant_split_data.begin(), constant_split_data.end());

                status.retcode = dq_split_kernel->Init(options);
                if (ppl::common::RC_SUCCESS != status.retcode) {
                    LOG(ERROR) << "init kernel[" << dq_split_kernel->GetNode()->GetName()
                        << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
                    return status;
                }
                options.partition_info->kernels[split_node->GetId()].reset(dq_split_kernel);
                LOG(DEBUG) << "process fuse for Split[" << split_node->GetName() << "] success";
            }
        }
    }

    return status;
}

}}}}}
