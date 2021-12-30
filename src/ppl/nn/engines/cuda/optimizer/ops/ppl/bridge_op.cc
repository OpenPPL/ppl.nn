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

#include "ppl/nn/engines/cuda/optimizer/ops/ppl/bridge_op.h"

#include "ppl/nn/engines/cuda/kernels/ppl/bridge_kernel.h"
#include "cudakernel/reformat/reformat.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode BridgeOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto& in_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto& out_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        auto in_edge_id = info->GetInput<TensorImpl>(0)->GetEdge()->GetId();
        auto& in_quant = quant->at(in_edge_id);
        if (in_shape.GetDataType() == ppl::common::DATATYPE_UNKNOWN && in_quant.type != ppl::common::DATATYPE_UNKNOWN) {
            in_shape.SetDataType(in_quant.type);
        }
        out_shape.SetDataType(in_shape.GetDataType());
        return RC_SUCCESS;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
            LOG(ERROR) << "1 input/output required.";
            return RC_INVALID_VALUE;
        }
        auto& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        info->GetOutput<TensorImpl>(0)->GetShape().Reshape(in_shape0.GetDims(), in_shape0.GetRealDimCount());
        return RC_SUCCESS;
    };

    return RC_SUCCESS;
}

RetCode BridgeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* BridgeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<BridgeKernel>();
}

RetCode BridgeOp::AddInternalBridgeNode(ir::Node* node, ir::Node* new_node, ir::Edge* edge, ir::Graph* graph) {
    auto topo = graph->topo.get();
    auto ret_pair = topo->AddEdge("Bridge_Edge_" + edge->GetName() + "_" + node->GetName());
    auto new_edge = ret_pair.first;

    edge->DelConsumer(node->GetId());
    edge->AddConsumer(new_node->GetId());
    new_node->AddInput(edge->GetId());
    new_node->AddOutput(new_edge->GetId());
    new_edge->SetProducer(new_node->GetId());
    new_edge->AddConsumer(node->GetId());
    auto size = node->ReplaceInput(edge->GetId(), new_edge->GetId());
    if (size == 0) {
        LOG(ERROR) << "Replace error";
        return RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

RetCode BridgeOp::AddFinalBridgeNode(ir::Node* node, ir::Node* new_node, ir::Edge* edge, ir::Graph* graph) {
    auto topo = graph->topo.get();
    auto ret_pair = topo->AddEdge("Bridge_Final_Edge_" + edge->GetName() + "_" + node->GetName());
    auto new_edge = ret_pair.first;

    edge->SetProducer(new_node->GetId());
    new_node->AddInput(new_edge->GetId());
    new_node->AddOutput(edge->GetId());
    new_edge->SetProducer(node->GetId());
    new_edge->AddConsumer(new_node->GetId());
    node->ReplaceOutput(edge->GetId(), new_edge->GetId());

    for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto node_id = it.Get();
        auto node = topo->GetNodeById(node_id);
        node->ReplaceInput(edge->GetId(), new_edge->GetId());
        new_edge->AddConsumer(node_id);
    }
    edge->ClearConsumer();

    return RC_SUCCESS;
}

RetCode BridgeOp::DeleteBridgeNode(ir::Node* node, ir::Graph* graph,
                                   std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                   std::vector<CudaTensorQuant>* quants) {
    auto topo = graph->topo.get();

    auto preedge_id = node->GetInput(0);
    auto postedge_id = node->GetOutput(0);

    if (topo->GetEdgeById(postedge_id)->CalcConsumerCount() == 0) { // final bridge node
        return RC_UNSUPPORTED;
    }

    auto nextnode_id = topo->GetEdgeById(postedge_id)->CreateConsumerIter().Get(); // consumer0
    auto prequant = quants->at(preedge_id);
    auto postquant = quants->at(postedge_id);

    auto preedge = topo->GetEdgeById(preedge_id);
    auto nextnode = topo->GetNodeById(nextnode_id);
    if (prequant.format == postquant.format && // two edge has the same format
        prequant.type == postquant.type && // two edge has the same type
        (prequant.type != DATATYPE_INT8 || EqualQuant(prequant, postquant)) && // two edge has the same quant
        topo->GetInput(topo->GetEdgeById(preedge_id)->GetName()) == INVALID_EDGEID && // and preedge is not graph input
        topo->GetExtraInput(topo->GetEdgeById(preedge_id)->GetName()) == INVALID_EDGEID && // and preedge is not graph extrainput
        topo->GetOutput(topo->GetEdgeById(postedge_id)->GetName()) == INVALID_EDGEID) { // and postedge is not graph output

        preedge->DelConsumer(node->GetId());
        preedge->AddConsumer(nextnode_id);
        nextnode->ReplaceInput(postedge_id, preedge_id);

        topo->DelEdgeById(postedge_id);
        topo->DelNodeById(node->GetId());

        return RC_SUCCESS;
    } else {
        // LOG(ERROR) << "node name " << node->GetName();
        // LOG(ERROR) << "format " << prequant.format << " " << postquant.format;
        // LOG(ERROR) << "type " << prequant.type << " " << postquant.type;
        // LOG(ERROR) << "quant " << EqualQuant(prequant, postquant);
    }
    return RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::cuda
