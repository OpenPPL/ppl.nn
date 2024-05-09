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

#include "ppl/nn/params/opmx/column_parallel_linear_param.h"

#include "ppl/nn/engines/llm_cuda/ops/opmx/i4f16/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i4f16/row_parallel_linear_op.h"

#include "ppl/kernel/llm/cuda/pmx/i4f16/quantize.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace i4f16 {

static std::string GetScaleName(const std::string& tensor_name) {
    return tensor_name + ".scale";
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

    const int64_t out_features_pack_size = 4;
    const int64_t weight_quant_group_size = 128;

    if (out_features % out_features_pack_size != 0) {
        LOG(WARNING) << "only support out_features(" << out_features << ") aligned with " << out_features_pack_size;
        return {ppl::common::RC_SUCCESS, nullptr};
    }

    // N must be aligned to 128 for int4 gemm api
    if (out_features % weight_quant_group_size != 0) {
        LOG(WARNING) << "only support out_features(" << out_features << ") aligned with " << weight_quant_group_size;
        return {ppl::common::RC_SUCCESS, nullptr};
    }

    if (in_features % weight_quant_group_size != 0) {
        LOG(WARNING) << "only support in_features(" << in_features << ") aligned with " << weight_quant_group_size;
        return {ppl::common::RC_SUCCESS, nullptr};
    }

    // check wether this weight has been processed
    if (loaded_constants->find(weight_edge->GetId()) != loaded_constants->end()) {
        auto scale_edge = topo->GetEdge(scale_name);
        return {ppl::common::RC_SUCCESS, scale_edge};
    }

    auto weight_shape = &shapes->at(weight_edge->GetId());
    if (weight_shape->data_type != ppl::common::DATATYPE_FLOAT16) {
        LOG(WARNING) << "only support i4f16 quantize for fp16 weight";
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
    quantized_weight_buffer.GetShape()->Reshape({out_features / out_features_pack_size, in_features});
    quantized_weight_buffer.GetShape()->SetDataType(ppl::common::DATATYPE_INT4X4);
    quantized_weight_buffer.GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    quantized_weight_buffer.SetDevice(options.device);
    rc = quantized_weight_buffer.ReallocBuffer();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "realloc buffer for quantize weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return {ppl::common::RC_OUT_OF_MEMORY, nullptr};
    }

    // alloc buffer for scale
    RuntimeConstantInfo scale_buffer;
    scale_buffer.GetShape()->Reshape({in_features / weight_quant_group_size, out_features});
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
    weight_buffer.GetShape()->Reshape({out_features, in_features});
    weight_buffer.GetShape()->SetDataType(weight_shape->data_type);
    weight_buffer.GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
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

    rc = ppl::kernel::llm::cuda::pmx::i4f16::minmax_quantize_fp16(
        options.device->GetStream(),
        weight_buffer.GetBufferPtr(),
        out_features,
        in_features,
        weight_quant_group_size,
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
    scale_shape->dims = {in_features / weight_quant_group_size, out_features};

    // change weight shape and datatype
    weight_shape->data_type = ppl::common::DATATYPE_INT4X4;
    weight_shape->dims = {out_features / out_features_pack_size, in_features};

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

static ppl::common::RetCode QuantizeLinearWeightOnly(
    ir::Node* linear_node,
    const OptKernelOptions& options,
    const int64_t in_features,
    const int64_t out_features)
{
    auto topo = options.graph->topo.get();

    auto weight_edge = topo->GetEdge(linear_node->GetInput(1));
    auto weight_scale_edge = topo->GetEdge(GetScaleName(weight_edge->GetName()));
    if (weight_scale_edge == nullptr) {
        LOG(ERROR) << "scale edge[" << GetScaleName(weight_edge->GetName()) << "] not found";
        return ppl::common::RC_NOT_FOUND;
    }

    { 
        // rearrange node and edge
        // before: input_edge --> linear_node -> output_edge
        //         weight_edge -|
        //           bias_edge -/
        // after: input_edge --> linear_node -> output_edge
        //        weight_edge -|
        //  weight_scale_edge -|
        //          bias_edge -/
        linear_node->InsertInput(2, weight_scale_edge->GetId());
        weight_scale_edge->AddConsumer(linear_node->GetId());
    }

    return ppl::common::RC_SUCCESS;
}

static OptPassStatus QuantizeColunmParallelLinear(ir::Node* linear_node, const OptKernelOptions& options) {
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    auto param = std::static_pointer_cast<ppl::nn::opmx::ColumnParallelLinearParam>(options.graph->data->attrs[linear_node->GetId()]);
    const auto in_features = param->in_features;
    const auto out_features_per_part = param->out_features / options.device->GetTensorParallelNcclParam()->size;
    if ((in_features % 32 != 0) || (out_features_per_part % 32 != 0 )) {
        LOG(WARNING) << "in_features and out_features_per_part should be aligned with 32 for i4f16 quantization, "
            <<"ColumnParallelLinear[" << linear_node->GetName() << "], whose weight is ("
            << out_features_per_part << ", " << in_features << ") will not be quantized";
        return status;
    }

    {
        LOG(DEBUG) << "processing i4f16 for ColumnParallelLinear[" << linear_node->GetName() << "]";
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
        status.retcode = QuantizeLinearWeightOnly(linear_node, options, in_features, out_features_per_part);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            return status;
        }
    }

    if (status.graph_modified) {
        // change ColunmParallelLinear to i4f16.ColunmParallelLinear
        linear_node->SetType({"opmx.i4f16", "ColumnParallelLinear", 1});
        auto q_linear_kernel = new opmx::I4F16ColumnParallelLinearOp(linear_node);
        status.retcode = q_linear_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            LOG(ERROR) << "init kernel[" << q_linear_kernel->GetNode()->GetName()
                << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
            return status;
        }
        options.partition_info->kernels[linear_node->GetId()].reset(q_linear_kernel);
        LOG(DEBUG) << "process i4f16 for ColumnParallelLinear[" << linear_node->GetName() << "] success";
    }

    return status;
}

static OptPassStatus QuantizeRowParallelLinear(ir::Node* linear_node, const OptKernelOptions& options) {
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    auto param = std::static_pointer_cast<ppl::nn::opmx::RowParallelLinearParam>(options.graph->data->attrs[linear_node->GetId()]);
    const auto in_features_per_part = param->in_features / options.device->GetTensorParallelNcclParam()->size;
    const auto out_features = param->out_features;
    if ((in_features_per_part % 32 != 0) || (out_features % 32 != 0 )) {
        LOG(WARNING) << "in_features_per_part and out_features should be aligned with 32 for i4f16 quantization, "
            <<"ColumnParallelLinear[" << linear_node->GetName() << "], whose weight is ("
            << out_features << ", " << in_features_per_part << ") will not be quantized";
        return status;
    }

    {
        LOG(DEBUG) << "processing i4f16 for RowParallelLinear[" << linear_node->GetName() << "]";
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
        status.retcode = QuantizeLinearWeightOnly(linear_node, options, in_features_per_part, out_features);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            return status;
        }
    }

    if (status.graph_modified) {
        // change RowParallelLinear to i4f16.RowParallelLinear
        linear_node->SetType({"opmx.i4f16", "RowParallelLinear", 1});
        auto q_linear_kernel = new opmx::I4F16RowParallelLinearOp(linear_node);
        status.retcode = q_linear_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            LOG(ERROR) << "init kernel[" << q_linear_kernel->GetNode()->GetName()
                << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
            return status;
        }
        options.partition_info->kernels[linear_node->GetId()].reset(q_linear_kernel);
        LOG(DEBUG) << "process i4f16 for RowParallelLinear[" << linear_node->GetName() << "] success";
    }

    return status;
}

OptPassStatus QuantizationPass(const OptKernelOptions& options)
{
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    if (options.device->GetSMVersion() < 80) {
        LOG(WARNING) << "i4f16 quantize only support sm >= 80 now";
        return status;
    }

    for (auto it = options.graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "opmx" && node->GetType().name == "ColumnParallelLinear") {
            auto ret = QuantizeColunmParallelLinear(node, options);
            status.graph_modified = status.graph_modified || ret.graph_modified;
            status.retcode = ret.retcode;
            if (ppl::common::RC_SUCCESS != status.retcode)
                return status;
        }
        if (node->GetType().domain == "opmx" && node->GetType().name == "RowParallelLinear") {
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
