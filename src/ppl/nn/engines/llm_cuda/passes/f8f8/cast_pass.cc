#ifdef PPLNN_ENABLE_FP8

#include "cast_pass.h"

#include "ppl/nn/params/opmx/column_parallel_linear_param.h"

#include "ppl/nn/engines/llm_cuda/ops/opmx/f8f8/online_cast_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/f8f8/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/f8f8/row_parallel_linear_op.h"

#include "ppl/kernel/llm/cuda/pmx/f8f8/cast.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace f8f8 {

static std::string GetCastedEdgeName(const std::string& tensor_name) {
    return tensor_name + ".casted";
}

static std::string GetCastNodeName(const std::string& tensor_name) {
    return "Cast." + tensor_name;
}

struct CastWeightResult final {
    ppl::common::RetCode retcode;
    bool casted;
};


static CastWeightResult CastWeight(
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

    std::set<std::string> consumer_white_list = {
        "ColumnParallelLinear",
        "RowParallelLinear",
    };

    for (auto iter = weight_edge->CreateConsumerIter(); iter.IsValid(); iter.Forward()) {
        auto &consumer_type = topo->GetNode(iter.Get())->GetType();
        if (consumer_white_list.find(consumer_type.name) == consumer_white_list.end()) {
            LOG(WARNING) << "failed to f8f8 cast weight[" << weight_edge->GetName() << "], "
                << "met unsupported consumer type [" << consumer_type.domain << ":" << consumer_type.name << "]";
            return {ppl::common::RC_SUCCESS, false};
        }
    }

    // check wether this weight has been processed
    if (loaded_constants->find(weight_edge->GetId()) != loaded_constants->end()) {
        return {ppl::common::RC_SUCCESS, true};
    }

    auto weight_shape = &shapes->at(weight_edge->GetId());

    if (weight_shape->data_type != ppl::common::DATATYPE_FLOAT16) {
        LOG(WARNING) << "only support f8f8 cast for fp16 weight";
        return {ppl::common::RC_SUCCESS, false};
    }

    ppl::common::RetCode rc;

    // alloc buffer for casted weight
    RuntimeConstantInfo casted_weight_buffer;
    casted_weight_buffer.GetShape()->Reshape({out_features, in_features});
    casted_weight_buffer.GetShape()->SetDataType(ppl::common::DATATYPE_FLOAT8E4M3);
    casted_weight_buffer.GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    casted_weight_buffer.SetDevice(options.device);
    rc = casted_weight_buffer.ReallocBuffer();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "realloc buffer for cast weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return {ppl::common::RC_OUT_OF_MEMORY, false};
    }

    // alloc buffer for origin weight at last
    // NOTE: it must be alloced at last to avoid memory fragmentation when it freed after being casted 
    RuntimeConstantInfo weight_buffer;
    weight_buffer.Reshape(*casted_weight_buffer.GetShape());
    weight_buffer.GetShape()->SetDataType(weight_shape->data_type);
    weight_buffer.SetDevice(options.device);

    // use zero copy to reduce GPU memory fragmentation
    void* weight_pinned_host_buffer = nullptr;
    auto cuda_err = cudaMallocHost(&weight_pinned_host_buffer, weight_buffer.GetShape()->CalcBytesIncludingPadding(), cudaHostAllocMapped);
    if (cudaSuccess != cuda_err) {
        LOG(ERROR) << "realloc pinned buffer for weight[" << weight_edge->GetName() << "] failed: " << cudaGetErrorString(cuda_err);
        return {ppl::common::RC_OUT_OF_MEMORY, false};
    }
    void *weight_pinned_dev_buffer = nullptr;
    cuda_err = cudaHostGetDevicePointer(&weight_pinned_dev_buffer, weight_pinned_host_buffer, 0);
    if (cudaSuccess != cuda_err) {
        LOG(ERROR) << "get device pinned buffer for weight[" << weight_edge->GetName() << "] failed: " << cudaGetErrorString(cuda_err);
        return {ppl::common::RC_DEVICE_MEMORY_ERROR, false};
    }
    weight_buffer.SetBuffer(weight_pinned_dev_buffer);

    // copy fp16 data to pinned memory for cast
    auto weight_host = &constants->at(weight_edge->GetId());
    memcpy(weight_pinned_host_buffer, weight_host->data.GetData(), weight_host->data.GetSize());
    constants->erase(weight_edge->GetId());

    // call cast kernel here
    rc = ppl::kernel::llm::cuda::pmx::f8f8::cast_fp16(
        options.device->GetStream(),
        weight_buffer.GetBufferPtr(),
        out_features,
        in_features,
        casted_weight_buffer.GetBufferPtr());
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "do cast for weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return {ppl::common::RC_DEVICE_RUNTIME_ERROR, false};
    }
    rc = options.device->Synchronize();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "synchronize cast for weight[" << weight_edge->GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return {ppl::common::RC_DEVICE_RUNTIME_ERROR, false};
    }

    // change weight datatype
    weight_shape->data_type = ppl::common::DATATYPE_FLOAT8E4M3;

    // emplace GPU buffer to runtime constants
    loaded_constants->emplace(weight_edge->GetId(), std::move(casted_weight_buffer));

    cuda_err = cudaFreeHost(weight_pinned_host_buffer);
    if (cudaSuccess != cuda_err) {
        LOG(ERROR) << "free pinned buffer for weight[" << weight_edge->GetName() << "] failed: " << cudaGetErrorString(cuda_err);
        return {ppl::common::RC_DEVICE_MEMORY_ERROR, false};
    }

    return {ppl::common::RC_SUCCESS, true};
}


static ppl::common::RetCode CastLinear(
    ir::Node* linear_node,
    const OptKernelOptions& options,
    const int64_t in_features,
    const int64_t out_features)
{
    auto topo = options.graph->topo.get();
    auto kernels = &options.partition_info->kernels;

    auto input_edge = topo->GetEdge(linear_node->GetInput(0));

    // sometime there are 2 linear node consume same input,
    // such as llama's FeedForward: y = w2(silu(w1(x)) * w3(x)).
    // q_input_exits is for checking if input of linear has been casted.
    auto casted_input_exits = false;
    auto casted_input_name = GetCastedEdgeName(input_edge->GetName());
    auto edge_ret_pair = topo->AddEdge(casted_input_name);
    if (!edge_ret_pair.second) {
        LOG(DEBUG) << "casted edge[" << casted_input_name << "] for input[" << input_edge->GetName() << "] exists";
        casted_input_exits = true;
    } else {
        LOG(DEBUG) << "add casted edge[" << casted_input_name << "] for input[" << input_edge->GetName() << "] success";
    }
    auto casted_input_edge = edge_ret_pair.first;

    auto cast_node_exists = false;
    auto cast_node_name = GetCastNodeName(input_edge->GetName());
    auto node_ret_pair = topo->AddNode(cast_node_name);
    auto cast_node = node_ret_pair.first;
    if (!node_ret_pair.second) {
        // we shoud check the cast method of cast_node
        if (cast_node->GetType().domain != "opmx.f8f8" || cast_node->GetType().name != "OnlineCast") {
            LOG(ERROR) << "cast node[" << cast_node_name << "] for input[" << input_edge->GetName() << "] exists, "
                << "expect for [opmx.f8f8:OnlineCast] but given ["
                << cast_node->GetType().domain << ":" << cast_node->GetType().name << "]";
            return ppl::common::RC_EXISTS;
        }
        LOG(DEBUG) << "cast node[" << cast_node_name << "] for input[" << input_edge->GetName() << "] exists";
        cast_node_exists = true;
    } else {
        LOG(DEBUG) << "add cast node[" << cast_node_name << "] for input[" << input_edge->GetName() << "] success";
        cast_node->SetType({"opmx.f8f8", "OnlineCast", 1});
    }

    bool input_has_casted = cast_node_exists && casted_input_exits;
    if (!input_has_casted && (cast_node_exists || casted_input_exits)) {
        LOG(ERROR) << "input[" << input_edge->GetName() << "] has not been completely f8f8 casted: "
            << "cast_node(" << cast_node_exists << "), "
            << "cast_input(" << casted_input_exits << ")";
    }

    { 
        // rearrange node and edge
        // before: input_edge --> linear_node -> output_edge
        //         weight_edge -> |
        // after: input_edge -> cast_node -> casted_input_edge ---> linear_node -> output_edge
        //                             |  casted_weight_edge -/
        //                                        bias_edge -/
        linear_node->ReplaceInput(input_edge->GetId(), casted_input_edge->GetId());

        if (!input_has_casted) {
            cast_node->AddInput(input_edge->GetId());
            cast_node->AddOutput(casted_input_edge->GetId());
        }

        input_edge->DelConsumer(linear_node->GetId());
        input_edge->AddConsumer(cast_node->GetId());

        casted_input_edge->AddConsumer(linear_node->GetId());
        casted_input_edge->SetProducer(cast_node->GetId());
    }

    {
        if (!input_has_casted) {
            auto cast_kernel = std::unique_ptr<LlmCudaOptKernel>(new opmx::F8F8OnlineCastOp(cast_node));
            auto rc = cast_kernel->Init(options);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "init kernel[" << cast_kernel->GetNode()->GetName() << " failed: " << ppl::common::GetRetCodeStr(rc);
                return rc;
            }
            kernels->emplace(cast_node->GetId(), std::move(cast_kernel));
        }
    }

    return ppl::common::RC_SUCCESS;  
}


static OptPassStatus CastColunmParallelLinear(ir::Node* linear_node, const OptKernelOptions& options) {
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    auto param = std::static_pointer_cast<ppl::nn::opmx::ColumnParallelLinearParam>(options.graph->data->attrs[linear_node->GetId()]);
    const auto in_features = param->in_features;
    const auto out_features_per_part = param->out_features / options.device->GetTensorParallelNcclParam()->size;
    if ((in_features % 16 != 0) || (out_features_per_part % 16 != 0 )) {
        LOG(WARNING) << "in_features and out_features_per_part should be aligned with 16 for f8f8 cast, "
            <<"ColumnParallelLinear[" << linear_node->GetName() << "], whose weight is ("
            << out_features_per_part << ", " << in_features << ") will not be casted";
        return status;
    }
    

    {
        LOG(DEBUG) << "processing f8f8 for ColumnParallelLinear[" << linear_node->GetName() << "]";
        auto cast_ret = CastWeight(linear_node, options, in_features, out_features_per_part);
        if (cast_ret.retcode != ppl::common::RC_SUCCESS) {
            status.retcode = cast_ret.retcode;
            status.graph_modified = true;
            return status;
        }
        if (cast_ret.casted == false) {
            return status;
        }

        status.graph_modified = true;
        status.retcode = CastLinear(linear_node, options, in_features, out_features_per_part);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            return status;
        }
    }

    if (status.graph_modified) {
        // change ColunmParallelLinear to f8f8.ColunmParallelLinear
        linear_node->SetType({"opmx.f8f8", "ColumnParallelLinear", 1});
        auto cast_linear_kernel = new opmx::F8F8ColumnParallelLinearOp(linear_node);
        status.retcode = cast_linear_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            LOG(ERROR) << "init kernel[" << cast_linear_kernel->GetNode()->GetName()
                << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
            return status;
        }
        options.partition_info->kernels[linear_node->GetId()].reset(cast_linear_kernel);
        LOG(DEBUG) << "process f8f8 for ColumnParallelLinear[" << linear_node->GetName() << "] success";
    }

    return status;
}

static OptPassStatus CastRowParallelLinear(ir::Node* linear_node, const OptKernelOptions& options) {
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    auto param = std::static_pointer_cast<ppl::nn::opmx::RowParallelLinearParam>(options.graph->data->attrs[linear_node->GetId()]);
    const auto in_features_per_part = param->in_features / options.device->GetTensorParallelNcclParam()->size;
    const auto out_features = param->out_features;
    if ((in_features_per_part % 16 != 0) || (out_features % 16 != 0 )) {
        LOG(WARNING) << "in_features_per_part and out_features should be aligned with 16 for f8f8 cast, "
            <<"ColumnParallelLinear[" << linear_node->GetName() << "], whose weight is ("
            << out_features << ", " << in_features_per_part << ") will not be casted";
        return status;
    }

    {
        LOG(DEBUG) << "processing f8f8 for RowParallelLinear[" << linear_node->GetName() << "]";
        auto cast_ret = CastWeight(linear_node, options, in_features_per_part, out_features);
        if (cast_ret.retcode != ppl::common::RC_SUCCESS) {
            status.retcode = cast_ret.retcode;
            status.graph_modified = true;
            return status;
        }
        if (cast_ret.casted == false) {
            return status;
        }

        status.graph_modified = true;
        status.retcode = CastLinear(linear_node, options, in_features_per_part, out_features);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            return status;
        }
    }

    if (status.graph_modified) {
        // change RowParallelLinear to f8f8.RowParallelLinear
        linear_node->SetType({"opmx.f8f8", "RowParallelLinear", 1});
        auto cast_linear_kernel = new opmx::F8F8RowParallelLinearOp(linear_node);
        status.retcode = cast_linear_kernel->Init(options);
        if (ppl::common::RC_SUCCESS != status.retcode) {
            LOG(ERROR) << "init kernel[" << cast_linear_kernel->GetNode()->GetName()
                << " failed: " << ppl::common::GetRetCodeStr(status.retcode);
            return status;
        }
        options.partition_info->kernels[linear_node->GetId()].reset(cast_linear_kernel);
        LOG(DEBUG) << "process f8f8 for RowParallelLinear[" << linear_node->GetName() << "] success";
    }

    return status;
}

OptPassStatus CastPass(const OptKernelOptions& options)
{
    OptPassStatus status = {ppl::common::RC_SUCCESS, false};

    if (options.device->GetSMVersion() < 89) {
        LOG(WARNING) << "f8f8 cast only support sm >= 89 now";
        return status;
    }

    for (auto it = options.graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "opmx" && node->GetType().name == "ColumnParallelLinear") {
            auto ret = CastColunmParallelLinear(node, options);
            status.graph_modified = status.graph_modified || ret.graph_modified;
            status.retcode = ret.retcode;
            if (ppl::common::RC_SUCCESS != status.retcode)
                return status;
        }
        if (node->GetType().domain == "opmx" && node->GetType().name == "RowParallelLinear") {
            auto ret = CastRowParallelLinear(node, options);
            status.graph_modified = status.graph_modified || ret.graph_modified;
            status.retcode = ret.retcode;
            if (ppl::common::RC_SUCCESS != status.retcode)
                return status;
        }
    }

    return status;
}

}}}}}

#endif