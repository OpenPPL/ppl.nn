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

#include "quantization_pass.h"

#include "ppl/nn/params/pmx/column_parallel_linear_param.h"

#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/online_quantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/online_dequantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/row_parallel_linear_op.h"

#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace i8i8 {

static std::string GetScaleName(const std::string& tensor_name) {
    return tensor_name + ".scale";
}

static std::string GetQuantizedEdgeName(const std::string& tensor_name) {
    return tensor_name + ".quantized";
}

static std::string GetQuantizeNodeName(const std::string& tensor_name) {
    return "Quant." + tensor_name;
}

static std::string GetDequantizeNodeName(const std::string& tensor_name) {
    return "Dequant." + tensor_name;
}

struct QuantizeWeightResult final {
    ppl::common::RetCode retcode;
    ir::Edge* scale_edge;
};

// return scale edge
static QuantizeWeightResult QuantizeWeight(
    ir::Node* linear_node,
    const OptKernelOptions& options,
    const int64_t in_features,
    const int64_t out_features)
{
    auto topo = options.graph->topo.get();
    auto constants = &options.graph->data->constants;
    auto shapes = &options.graph->data->shapes;
    auto loaded_constants = &options.partition_info->constants;

    auto weight_edge = topo->GetEdge(linear_node->GetInput(1));
    auto scale_name = GetScaleName(weight_edge->GetName());

    std::set<std::string> consumer_white_list = {
        "ColumnParallelLinear",
        "RowParallelLinear",
    };

    for (auto iter = weight_edge->CreateConsumerIter(); iter.IsValid(); iter.Forward()) {
        auto &consumer_type = topo->GetNode(iter.Get())->GetType();
        if (consumer_white_list.find(consumer_type.name) == consumer_white_list.end()) {
            LOG(WARNING) << "failed to i4f16 quantize weight[" << weight_edge->GetName() << "], "
                << "met unsupported consumer type [" << consumer_type.domain << ":" << consumer_type.name << "]";
            return {ppl::common::RC_SUCCESS, nullptr};
        }
    }

    // check wether this weight has been processed
    if (loaded_constants->find(weight_edge->GetId()) != loaded_constants->end()) {
        auto scale_edge = topo->GetEdge(scale_name);
        return {ppl::common::RC_SUCCESS, scale_edge};
    }

    auto weight_shape = &shapes->at(weight_edge->GetId());

    if (weight_shape->data_type != ppl::common::DATATYPE_FLOAT16) {
        LOG(WARNING) << "only support i8i8 quantize for fp16 weight";
        return {ppl::common::RC_SUCCESS, nullptr};
    }

    // add constant scale edge and check
    auto ret_pair = topo->AddEdge(scale_name);
    if (!ret_pair.second) {
        LOG(ERROR) << "add scale edge[" << scale_name << "] for weight[" << weight_edge->GetName() << "] failed";
        return {ppl::common::RC_EXISTS, nullptr};
    }
    LOG(DEBUG) << "add scale edge[" << scale_name << "] for weight[" << weight_edge->GetName() << "] success";
    auto scale_edge = ret_pair.first;
    topo->MarkAsConstant(scale_edge->GetId());

    ppl::common::RetCode rc;

    // alloc buffer for quantized weight
    RuntimeConstantInfo quantized_weight_buffer;
    quantized_weight_buffer.GetShape()->Reshape({out_features, in_features});
    quantized_weight_buffer.GetShape()->SetDataType(ppl::common::DATATYPE_INT8);
    quantized_weight_buffer.GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    quantized_weight_buffer.SetDevice(options.device);
    rc = quantized_weight_buffer.ReallocBuffer();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "realloc buffer for quantize weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return {ppl::common::RC_OUT_OF_MEMORY, nullptr};
    }

    // alloc buffer for scale
    RuntimeConstantInfo scale_buffer;
    scale_buffer.GetShape()->Reshape({out_features});
    scale_buffer.GetShape()->SetDataType(weight_shape->data_type);
    scale_buffer.GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    scale_buffer.SetDevice(options.device);
    rc = scale_buffer.ReallocBuffer();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "realloc buffer for scale of weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return {ppl::common::RC_OUT_OF_MEMORY, nullptr};
    }

    // alloc buffer for origin weight at last
    // NOTE: it must be alloced at last to avoid memory fragmentation when it freed after being quantized 
    RuntimeConstantInfo weight_buffer;
    weight_buffer.Reshape(*quantized_weight_buffer.GetShape());
    weight_buffer.GetShape()->SetDataType(weight_shape->data_type);
    weight_buffer.SetDevice(options.device);

    // use zero copy to reduce GPU memory fragmentation
    void* weight_pinned_host_buffer = nullptr;
    auto cuda_err = cudaMallocHost(&weight_pinned_host_buffer, weight_buffer.GetShape()->CalcBytesIncludingPadding(), cudaHostAllocMapped);
    if (cudaSuccess != cuda_err) {
        LOG(ERROR) << "realloc pinned buffer for weight[" << weight_edge->GetName() << "] failed: " << cudaGetErrorString(cuda_err);
        return {ppl::common::RC_OUT_OF_MEMORY, nullptr};
    }
    void *weight_pinned_dev_buffer = nullptr;
    cuda_err = cudaHostGetDevicePointer(&weight_pinned_dev_buffer, weight_pinned_host_buffer, 0);
    if (cudaSuccess != cuda_err) {
        LOG(ERROR) << "get device pinned buffer for weight[" << weight_edge->GetName() << "] failed: " << cudaGetErrorString(cuda_err);
        return {ppl::common::RC_DEVICE_MEMORY_ERROR, nullptr};
    }
    weight_buffer.SetBuffer(weight_pinned_dev_buffer);

    // copy fp16 data to pinned memory for quantize
    auto weight_host = &constants->at(weight_edge->GetId());
    memcpy(weight_pinned_host_buffer, weight_host->data.GetData(), weight_host->data.GetSize());
    constants->erase(weight_edge->GetId());

    // call quantize kernel here
    auto weight_layout = options.engine_options->cublas_layout_hint == CUBLAS_LAYOUT_AMPERE
        ? ppl::kernel::llm::cuda::MATRIX_LAYOUT_COL32_2R_4R4
        : ppl::kernel::llm::cuda::MATRIX_LAYOUT_ROW_MAJOR;
    rc = ppl::kernel::llm::cuda::pmx::i8i8::minmax_quantize_fp16(
        options.device->GetStream(),
        weight_buffer.GetBufferPtr(),
        out_features,
        in_features,
        ppl::kernel::llm::cuda::pmx::i8i8::hidden_up_scale,
        weight_layout,
        quantized_weight_buffer.GetBufferPtr(),
        scale_buffer.GetBufferPtr());
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "do quantize for weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        topo->DelEdge(scale_edge->GetId());
        return {ppl::common::RC_DEVICE_RUNTIME_ERROR, nullptr};
    }
    rc = options.device->Synchronize();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "synchronize quantize for weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        topo->DelEdge(scale_edge->GetId());
        return {ppl::common::RC_DEVICE_RUNTIME_ERROR, nullptr};
    }

    // fill constant scale shape
    auto scale_shape = &shapes->emplace(scale_edge->GetId(), std::move(ir::Shape())).first->second;
    scale_shape->data_type = weight_shape->data_type;
    scale_shape->data_format = ppl::common::DATAFORMAT_NDARRAY;
    scale_shape->dims = {out_features};

    // change weight datatype
    weight_shape->data_type = ppl::common::DATATYPE_INT8;

    // emplace GPU buffer to runtime constants
    loaded_constants->emplace(weight_edge->GetId(), std::move(quantized_weight_buffer));
    loaded_constants->emplace(scale_edge->GetId(), std::move(scale_buffer));

    cuda_err = cudaFreeHost(weight_pinned_host_buffer);
    if (cudaSuccess != cuda_err) {
        LOG(ERROR) << "free pinned buffer for weight[" << weight_edge->GetName() << "] failed: " << cudaGetErrorString(cuda_err);
        return {ppl::common::RC_DEVICE_MEMORY_ERROR, nullptr};
    }

    return {ppl::common::RC_SUCCESS, scale_edge};
}

static ppl::common::RetCode QuantizeLinear(
    ir::Node* linear_node,
    const OptKernelOptions& options,
    const int64_t in_features,
    const int64_t out_features,
    const bool bias_term)
{
    auto topo = options.graph->topo.get();
    auto kernels = &options.partition_info->kernels;

    auto weight_edge = topo->GetEdge(linear_node->GetInput(1));
    auto weight_scale_edge = topo->GetEdge(GetScaleName(weight_edge->GetName()));
    if (weight_scale_edge == nullptr) {
        LOG(ERROR) << "scale edge[" << GetScaleName(weight_edge->GetName()) << "] not found";
        return ppl::common::RC_NOT_FOUND;
    }

    auto input_edge = topo->GetEdge(linear_node->GetInput(0));
    auto output_edge = topo->GetEdge(linear_node->GetOutput(0));

    // sometime there are 2 linear node consume same input,
    // such as llama's FeedForward: y = w2(silu(w1(x)) * w3(x)).
    // q_input_exits is for checking if input of linear has been quantized.
    auto q_input_exits = false;
    auto q_input_name = GetQuantizedEdgeName(input_edge->GetName());
    auto edge_ret_pair = topo->AddEdge(q_input_name);
    if (!edge_ret_pair.second) {
        LOG(DEBUG) << "quantized edge[" << q_input_name << "] for input[" << input_edge->GetName() << "] exists";
        q_input_exits = true;
    } else {
        LOG(DEBUG) << "add quantized edge[" << q_input_name << "] for input[" << input_edge->GetName() << "] success";
    }
    auto q_input_edge = edge_ret_pair.first;

    auto input_scale_exits = false;
    auto input_scale_name = GetScaleName(input_edge->GetName());
    edge_ret_pair = topo->AddEdge(input_scale_name);
    if (!edge_ret_pair.second) {
        LOG(DEBUG) << "scale edge[" << input_scale_name << "] for input[" << input_edge->GetName() << "] exists";
        input_scale_exits = true;
    } else {
        LOG(DEBUG) << "add scale edge[" << input_scale_name << "] for input[" << input_edge->GetName() << "] success";
    }
    auto input_scale_edge = edge_ret_pair.first;

    auto q_output_name = GetQuantizedEdgeName(output_edge->GetName());
    edge_ret_pair = topo->AddEdge(q_output_name);
    if (!edge_ret_pair.second) {
        LOG(ERROR) << "add quantized edge[" << q_output_name << "] for output[" << output_edge->GetName() << "] failed";
        return ppl::common::RC_EXISTS;
    }
    LOG(DEBUG) << "add quantized edge[" << q_output_name << "] for output[" << output_edge->GetName() << "] success";
    auto q_output_edge = edge_ret_pair.first;

    auto q_node_exists = false;
    auto q_node_name = GetQuantizeNodeName(input_edge->GetName());
    auto node_ret_pair = topo->AddNode(q_node_name);
    auto q_node = node_ret_pair.first;
    if (!node_ret_pair.second) {
        // we shoud check the quantization method of q_node
        if (q_node->GetType().domain != "pmx.i8i8" || q_node->GetType().name != "OnlineQuantize") {
            LOG(ERROR) << "quantize node[" << q_node_name << "] for input[" << input_edge->GetName() << "] exists, "
                << "expect for [pmx.i8i8:OnlineQuantize] but given ["
                << q_node->GetType().domain << ":" << q_node->GetType().name << "]";
            return ppl::common::RC_EXISTS;
        }
        LOG(DEBUG) << "quantize node[" << q_node_name << "] for input[" << input_edge->GetName() << "] exists";
        q_node_exists = true;
    } else {
        LOG(DEBUG) << "add quantize node[" << q_node_name << "] for input[" << input_edge->GetName() << "] success";
        q_node->SetType({"pmx.i8i8", "OnlineQuantize", 1});
    }

    auto dq_node_name = GetDequantizeNodeName(output_edge->GetName());
    node_ret_pair = topo->AddNode(dq_node_name);
    if (!node_ret_pair.second) {
        LOG(ERROR) << "add dequantize node[" << dq_node_name << "] for output[" << output_edge->GetName() << "] failed";
        return ppl::common::RC_EXISTS;
    }
    LOG(DEBUG) << "add dequantize node[" << dq_node_name << "] for output[" << output_edge->GetName() << "] success";
    auto dq_node = node_ret_pair.first;
    dq_node->SetType({"pmx.i8i8", "OnlineDequantize", 1});

    bool input_has_quantized = q_node_exists && q_input_exits && input_scale_exits;
    if (!input_has_quantized && (q_node_exists || q_input_exits || input_scale_exits)) {
        LOG(ERROR) << "input[" << input_edge->GetName() << "] has not been completely i8i8 quantized: "
            << "q_node(" << q_node_exists << "), "
            << "q_input(" << q_input_exits << "), "
            << "input_scale(" << input_scale_exits << ")";
    }

    { 
        // rearrange node and edge
        // before: input_edge --> linear_node -> output_edge
        //         weight_edge -> |
        // after: input_edge -> q_node -> q_input_edge ---> linear_node -> q_output_edge -> dq_node -> output_edge
        //                             |  q_weight_edge -/   |                           |
        //                             \  input_scale_edge  -|---------------------------/
        //                                weight_scale_edge -|
        //                                        bias_edge -/
        linear_node->ReplaceInput(input_edge->GetId(), q_input_edge->GetId());
        linear_node->ReplaceOutput(output_edge->GetId(), q_output_edge->GetId());
        linear_node->InsertInput(2, input_scale_edge->GetId());
        linear_node->InsertInput(3, weight_scale_edge->GetId());

        if (!input_has_quantized) {
            q_node->AddInput(input_edge->GetId());
            q_node->AddOutput(q_input_edge->GetId());
            q_node->AddOutput(input_scale_edge->GetId());
        }

        dq_node->AddInput(q_output_edge->GetId());
        dq_node->AddInput(input_scale_edge->GetId());
        dq_node->AddInput(weight_scale_edge->GetId());
        dq_node->AddOutput(output_edge->GetId());

        input_edge->DelConsumer(linear_node->GetId());
        input_edge->AddConsumer(q_node->GetId());

        q_input_edge->AddConsumer(linear_node->GetId());
        q_input_edge->SetProducer(q_node->GetId());

        input_scale_edge->AddConsumer(dq_node->GetId());
        input_scale_edge->AddConsumer(linear_node->GetId());
        input_scale_edge->SetProducer(q_node->GetId());

        weight_scale_edge->AddConsumer(dq_node->GetId());
        weight_scale_edge->AddConsumer(linear_node->GetId());

        output_edge->SetProducer(dq_node->GetId());

        q_output_edge->SetProducer(linear_node->GetId());
        q_output_edge->AddConsumer(dq_node->GetId());
    }

    if (bias_term) {
        auto bias_edge = topo->GetEdge(linear_node->GetInput(4));

        dq_node->AddInput(bias_edge->GetId());

        bias_edge->AddConsumer(dq_node->GetId());
    }

    {
        // opt kernels
        if (!input_has_quantized) {
            auto q_kernel = std::unique_ptr<LlmCudaOptKernel>(new pmx::I8I8OnlineQuantizeOp(q_node));
            auto rc = q_kernel->Init(options);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "init kernel[" << q_kernel->GetNode()->GetName() << " failed: " << ppl::common::GetRetCodeStr(rc);
                return rc;
            }
            kernels->emplace(q_node->GetId(), std::move(q_kernel));
        }

        auto dq_kernel = std::unique_ptr<LlmCudaOptKernel>(new pmx::I8I8OnlineDequantizeOp(dq_node));
        auto rc = dq_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != rc) {
            LOG(ERROR) << "init kernel[" << dq_kernel->GetNode()->GetName() << " failed: " << ppl::common::GetRetCodeStr(rc);
            return rc;
        }
        ((pmx::I8I8OnlineDequantizeOp*)(dq_kernel.get()))->GetParam()->bias_term = bias_term;
        kernels->emplace(dq_node->GetId(), std::move(dq_kernel));
    }

    return ppl::common::RC_SUCCESS;
}

static ppl::common::RetCode QuantizeLinearSelfDequant(
    ir::Node* linear_node,
    const OptKernelOptions& options,
    const int64_t in_features,
    const int64_t out_features)
{
    auto topo = options.graph->topo.get();
    auto kernels = &options.partition_info->kernels;

    auto weight_edge = topo->GetEdge(linear_node->GetInput(1));
    auto weight_scale_edge = topo->GetEdge(GetScaleName(weight_edge->GetName()));
    if (weight_scale_edge == nullptr) {
        LOG(ERROR) << "scale edge[" << GetScaleName(weight_edge->GetName()) << "] not found";
        return ppl::common::RC_NOT_FOUND;
    }

    auto input_edge = topo->GetEdge(linear_node->GetInput(0));

    // sometime there are 2 linear node consume same input,
    // such as llama's FeedForward: y = w2(silu(w1(x)) * w3(x)).
    // q_input_exits is for checking if input of linear has been quantized.
    auto q_input_exits = false;
    auto q_input_name = GetQuantizedEdgeName(input_edge->GetName());
    auto edge_ret_pair = topo->AddEdge(q_input_name);
    if (!edge_ret_pair.second) {
        LOG(DEBUG) << "quantized edge[" << q_input_name << "] for input[" << input_edge->GetName() << "] exists";
        q_input_exits = true;
    } else {
        LOG(DEBUG) << "add quantized edge[" << q_input_name << "] for input[" << input_edge->GetName() << "] success";
    }
    auto q_input_edge = edge_ret_pair.first;

    auto input_scale_exits = false;
    auto input_scale_name = GetScaleName(input_edge->GetName());
    edge_ret_pair = topo->AddEdge(input_scale_name);
    if (!edge_ret_pair.second) {
        LOG(DEBUG) << "scale edge[" << input_scale_name << "] for input[" << input_edge->GetName() << "] exists";
        input_scale_exits = true;
    } else {
        LOG(DEBUG) << "add scale edge[" << input_scale_name << "] for input[" << input_edge->GetName() << "] success";
    }
    auto input_scale_edge = edge_ret_pair.first;

    auto q_node_exists = false;
    auto q_node_name = GetQuantizeNodeName(input_edge->GetName());
    auto node_ret_pair = topo->AddNode(q_node_name);
    auto q_node = node_ret_pair.first;
    if (!node_ret_pair.second) {
        // we shoud check the quantization method of q_node
        if (q_node->GetType().domain != "pmx.i8i8" || q_node->GetType().name != "OnlineQuantize") {
            LOG(ERROR) << "quantize node[" << q_node_name << "] for input[" << input_edge->GetName() << "] exists, "
                << "expect for [pmx.i8i8:OnlineQuantize] but given ["
                << q_node->GetType().domain << ":" << q_node->GetType().name << "]";
            return ppl::common::RC_EXISTS;
        }
        LOG(DEBUG) << "quantize node[" << q_node_name << "] for input[" << input_edge->GetName() << "] exists";
        q_node_exists = true;
    } else {
        LOG(DEBUG) << "add quantize node[" << q_node_name << "] for input[" << input_edge->GetName() << "] success";
        q_node->SetType({"pmx.i8i8", "OnlineQuantize", 1});
    }

    bool input_has_quantized = q_node_exists && q_input_exits && input_scale_exits;
    if (!input_has_quantized && (q_node_exists || q_input_exits || input_scale_exits)) {
        LOG(ERROR) << "input[" << input_edge->GetName() << "] has not been completely i8i8 quantized: "
            << "q_node(" << q_node_exists << "), "
            << "q_input(" << q_input_exits << "), "
            << "input_scale(" << input_scale_exits << ")";
    }

    { 
        // rearrange node and edge
        // before: input_edge --> linear_node -> output_edge
        //         weight_edge -> |
        // after: input_edge -> q_node -> q_input_edge ---> linear_node -> output_edge
        //                             |  q_weight_edge -/   |
        //                             \  input_scale_edge  -|
        //                                weight_scale_edge -|
        //                                        bias_edge -/
        linear_node->ReplaceInput(input_edge->GetId(), q_input_edge->GetId());
        linear_node->InsertInput(2, input_scale_edge->GetId());
        linear_node->InsertInput(3, weight_scale_edge->GetId());

        if (!input_has_quantized) {
            q_node->AddInput(input_edge->GetId());
            q_node->AddOutput(q_input_edge->GetId());
            q_node->AddOutput(input_scale_edge->GetId());
        }

        input_edge->DelConsumer(linear_node->GetId());
        input_edge->AddConsumer(q_node->GetId());

        q_input_edge->AddConsumer(linear_node->GetId());
        q_input_edge->SetProducer(q_node->GetId());

        input_scale_edge->AddConsumer(linear_node->GetId());
        input_scale_edge->SetProducer(q_node->GetId());

        weight_scale_edge->AddConsumer(linear_node->GetId());
    }

    {
        // opt kernels
        if (!input_has_quantized) {
            auto q_kernel = std::unique_ptr<LlmCudaOptKernel>(new pmx::I8I8OnlineQuantizeOp(q_node));
            auto rc = q_kernel->Init(options);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "init kernel[" << q_kernel->GetNode()->GetName() << " failed: " << ppl::common::GetRetCodeStr(rc);
                return rc;
            }
            kernels->emplace(q_node->GetId(), std::move(q_kernel));
        }
    }

    return ppl::common::RC_SUCCESS;
}

static OptPassStatus QuantizeColunmParallelLinear(ir::Node* linear_node, const OptKernelOptions& options) {
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    auto param = std::static_pointer_cast<ppl::nn::pmx::ColumnParallelLinearParam>(options.graph->data->attrs[linear_node->GetId()]);
    const auto in_features = param->in_features;
    const auto out_features_per_part = param->out_features / options.device->GetTensorParallelNcclParam()->size;
    if ((in_features % 32 != 0) || (out_features_per_part % 32 != 0 )) {
        LOG(WARNING) << "in_features and out_features_per_part should be aligned with 32 for i8i8 quantization, "
            <<"ColumnParallelLinear[" << linear_node->GetName() << "], whose weight is ("
            << out_features_per_part << ", " << in_features << ") will not be quantized";
        return status;
    }

    {
        LOG(DEBUG) << "processing i8i8 for ColumnParallelLinear[" << linear_node->GetName() << "]";
        auto quantize_ret = QuantizeWeight(linear_node, options, in_features, out_features_per_part);
        if (quantize_ret.retcode != ppl::common::RC_SUCCESS) {
            status.retcode = quantize_ret.retcode;
            status.graph_modified = true;
            return status;
        }
        if (quantize_ret.scale_edge == nullptr) {
            return status;
        }

        status.graph_modified = true;
        if (param->gather_output == false) {
            status.retcode = QuantizeLinear(linear_node, options, in_features, out_features_per_part, param->bias_term);
            if (ppl::common::RC_SUCCESS != status.retcode) {
                return status;
            }
            param->bias_term = false;
        } else {
            status.retcode = QuantizeLinearSelfDequant(linear_node, options, in_features, out_features_per_part);
            if (ppl::common::RC_SUCCESS != status.retcode) {
                return status;
            }
        }
    }

    if (status.graph_modified) {
        // change ColunmParallelLinear to i8i8.ColunmParallelLinear
        linear_node->SetType({"pmx.i8i8", "ColumnParallelLinear", 1});
        auto q_linear_kernel = new pmx::I8I8ColumnParallelLinearOp(linear_node);
        status.retcode = q_linear_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            LOG(ERROR) << "init kernel[" << q_linear_kernel->GetNode()->GetName()
                << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
            return status;
        }
        options.partition_info->kernels[linear_node->GetId()].reset(q_linear_kernel);
        LOG(DEBUG) << "process i8i8 for ColumnParallelLinear[" << linear_node->GetName() << "] success";
    }

    return status;
}

static OptPassStatus QuantizeRowParallelLinear(ir::Node* linear_node, const OptKernelOptions& options) {
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    auto param = std::static_pointer_cast<ppl::nn::pmx::RowParallelLinearParam>(options.graph->data->attrs[linear_node->GetId()]);
    const auto in_features_per_part = param->in_features / options.device->GetTensorParallelNcclParam()->size;
    const auto out_features = param->out_features;
    if ((in_features_per_part % 32 != 0) || (out_features % 32 != 0 )) {
        LOG(WARNING) << "in_features_per_part and out_features should be aligned with 32 for i8i8 quantization, "
            <<"ColumnParallelLinear[" << linear_node->GetName() << "], whose weight is ("
            << out_features << ", " << in_features_per_part << ") will not be quantized";
        return status;
    }

    {
        LOG(DEBUG) << "processing i8i8 for RowParallelLinear[" << linear_node->GetName() << "]";
        auto quantize_ret = QuantizeWeight(linear_node, options, in_features_per_part, out_features);
        if (quantize_ret.retcode != ppl::common::RC_SUCCESS) {
            status.retcode = quantize_ret.retcode;
            status.graph_modified = true;
            return status;
        }
        if (quantize_ret.scale_edge == nullptr) {
            return status;
        }

        status.graph_modified = true;
        status.retcode = QuantizeLinearSelfDequant(linear_node, options, in_features_per_part, out_features);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            return status;
        }
    }

    if (status.graph_modified) {
        // change RowParallelLinear to i8i8.RowParallelLinear
        linear_node->SetType({"pmx.i8i8", "RowParallelLinear", 1});
        auto q_linear_kernel = new pmx::I8I8RowParallelLinearOp(linear_node);
        status.retcode = q_linear_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            LOG(ERROR) << "init kernel[" << q_linear_kernel->GetNode()->GetName()
                << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
            return status;
        }
        options.partition_info->kernels[linear_node->GetId()].reset(q_linear_kernel);
        LOG(DEBUG) << "process i8i8 for RowParallelLinear[" << linear_node->GetName() << "] success";
    }

    return status;
}

OptPassStatus QuantizationPass(const OptKernelOptions& options)
{
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    if (options.device->GetSMVersion() < 80) {
        LOG(WARNING) << "i8i8 quantize only support sm >= 80 now";
        return status;
    }

    for (auto it = options.graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "pmx" && node->GetType().name == "ColumnParallelLinear") {
            auto ret = QuantizeColunmParallelLinear(node, options);
            status.graph_modified = status.graph_modified || ret.graph_modified;
            status.retcode = ret.retcode;
            if (ppl::common::RC_SUCCESS != status.retcode)
                return status;
        }
        if (node->GetType().domain == "pmx" && node->GetType().name == "RowParallelLinear") {
            auto ret = QuantizeRowParallelLinear(node, options);
            status.graph_modified = status.graph_modified || ret.graph_modified;
            status.retcode = ret.retcode;
            if (ppl::common::RC_SUCCESS != status.retcode)
                return status;
        }
    }

    return status;
}

}}}}}
