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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_conv.h"

#include <chrono>

#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

static std::string GetConvShapeString(conv_param_t &conv_param)
{
    return std::string("b" + std::to_string(conv_param.in_num)  + \
                       "_c" + std::to_string(conv_param.num_chl) + \
                       "_d" + std::to_string(conv_param.num_flt) + \
                       "_g" + std::to_string(conv_param.num_grp) + \
                       "_h" + std::to_string(conv_param.in_height) + \
                       "_w" + std::to_string(conv_param.in_width) + \
                       "_r" + std::to_string(conv_param.flt_height) + \
                       "_s" + std::to_string(conv_param.flt_width) + \
                       "_p" + std::to_string(conv_param.pad_height) + \
                       "_q" + std::to_string(conv_param.pad_width) + \
                       "_u" + std::to_string(conv_param.stride_height) + \
                       "_v" + std::to_string(conv_param.stride_width) + \
                       "_y" + std::to_string(conv_param.hole_height) + \
                       "_x" + std::to_string(conv_param.hole_width) + \
                       "_");
}

void TuringHMMAImpgemm::DeleteAttrParam(void*& param) {
    delete (CudaConvParam*)param;
    return;
}

void TuringHMMAImpgemm::GetAttrParam(void*& param) const {
    if (param == nullptr)
        param = new CudaConvParam();
    *(CudaConvParam*)param = attr_param_;
    return;
}

bool TuringHMMAImpgemm::IsSupported(const ir::Node* node, const OptKernelOptions& options, dataformat_t input_format) const {
    uint32_t group = (reinterpret_cast<CudaConvParam*>(options.param))->param.group;
    // check if conv is depthwise
    auto tensor1 = options.tensors->find(node->GetInput(1))->second->GetShape();
    if (group == tensor1.GetDim(0) && tensor1.GetDim(1) == 1 && group != 1) {
        return false;
    }
    // check if conv is quantization
    auto quant0 = options.quants->at(node->GetInput(0));
    if (quant0.type == DATATYPE_INT8) {
        return false;
    }
    if (input_format != DATAFORMAT_NHWC8) {
        return false;
    }
    return true;
}

double TuringHMMAImpgemm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "TuringHMMAImpgemm";
    attr_param_.extra_param.algo_info.kernel_index = 5100;

    // If the node has selcted, return answer directly
    auto pair = selection_res_.find(node->GetId());
    if (pair != selection_res_.end()) {
        attr_param_.extra_param.algo_info.kernel_index = pair->second.kernel_index;
        attr_param_.extra_param.algo_info.splitk = pair->second.splitk;
        attr_param_.extra_param.algo_info.splitf = pair->second.splitf;
        return pair->second.timer;
    }

    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    auto shape_in0 = options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_in2 = TensorShape();
    auto shape_out = options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_.param, temp_conv_param);
    ConvertToEmptyFuseParam(temp_fuse_param);

    auto algo_info = options.algos->find(GetConvShapeString(temp_conv_param));
    if (algo_info != options.algos->end()) {
        attr_param_.extra_param.algo_info.kernel_index = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
        return 0.0f;
    }

    if (options.args->quick_select) {
        return 0.0f;
    }

    // input H or W is too small
    if (shape_in0.GetDim(2) + 2 * temp_conv_param.pad_height < shape_in1.GetDim(2) ||
        shape_in0.GetDim(3) + 2 * temp_conv_param.pad_width < shape_in1.GetDim(3)) {
        shape_in0.SetDim(2, shape_in1.GetDim(2));
        shape_in0.SetDim(3, shape_in1.GetDim(3));
    }

    // Padding
    shape_in0.SetDim(1, shape_in1.GetDim(1) * attr_param_.param.group);
    uint32_t k_per_grp = shape_in1.GetDim(0) / attr_param_.param.group;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;
    shape_in1.SetDim(0, k_per_grp_pad * attr_param_.param.group);
    if (temp_conv_param.has_bias) {
        shape_in2 = options.tensors->find(node->GetInput(2))->second->GetShape();
        shape_in2.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);
    }

    RetCode status;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(input_buffer, shape_in0.GetBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(weight_buffer, shape_in1.GetBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(bias_buffer, shape_in2.GetBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(output_buffer, shape_out.GetBytesIncludingPadding(), ALGO_MAX_TIME)

    uint64_t size = PPLCUDAConvolutionGetCompilationBufSize(shape_in0.GetDataType(), temp_conv_param);
    ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, size, ALGO_MAX_TIME)

    // Do select
    auto stream = options.device->GetStream();
    algo_param_t algo_param;
    PPLCUDAConvolutionSelectKernel(stream, shape_in0.GetDataType(), (int4*)input_buffer.addr, (int4*)weight_buffer.addr,
                                   (int4*)output_buffer.addr, (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
                                   algo_param, temp_conv_param, temp_fuse_param);

    attr_param_.extra_param.algo_info.kernel_index = algo_param.kid;
    attr_param_.extra_param.algo_info.splitk = algo_param.splitk;
    attr_param_.extra_param.algo_info.splitf = algo_param.splitf;

    auto run_begin_ts = std::chrono::system_clock::now();
    PPLCUDAConvolutionForwardImp(stream, shape_in0.GetDataType(), (int4*)input_buffer.addr, (int4*)weight_buffer.addr,
                                 (int4*)output_buffer.addr, (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
                                 algo_param, temp_conv_param, temp_fuse_param);
    auto run_end_ts = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    double timer = (double)diff.count() / 1000;

    LOG(DEBUG) << "Select TuringHMMAImpgemm algorithm with kernel index "
               << attr_param_.extra_param.algo_info.kernel_index << " and excute timer " << timer << " for node["
               << node->GetName() << "]";

    SelectionInfo temp_res(algo_param.kid, algo_param.splitk, algo_param.splitf, timer);
    selection_res_.emplace(node->GetId(), std::move(temp_res));
    return timer;
}

RetCode TuringHMMAImpgemm::ModifyParam(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdgeById(node->GetInput(1));
    auto weight_node = topo->GetNodeById(weight_edge->GetProducer());

    auto shape_in0 = options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_out = options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    RetCode status;
    conv_param_t temp_conv_param;
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_.param, temp_conv_param);

    uint32_t k_per_grp = shape_in1.GetDim(0) / temp_conv_param.num_grp;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;

    // Split weight format to group padding
    auto stream = options.device->GetStream();
    auto weight_iter = data->constants.find(weight_node->GetInput(0));
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(weight_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = weight_node->GetInput(0);
        auto postedge_id = node->GetInput(1);
        auto preshape = options.tensors->find(preedge_id)->second->GetShape();
        auto postshape = options.tensors->find(postedge_id)->second->GetShape();
        auto newshape = postshape;
        newshape.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);

        RuntimeConstantInfo weight_constat_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            weight_constat_info.Reshape(postshape); // give the init shape, but the actual shape is padded
            weight_constat_info.SetBuffer(buffer, options.device, true);
        }

        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, postshape.GetBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = options.device->GetDataConverter()->ConvertFromHost(&temp_buffer, postshape,
                                                                     weight_iter->second.data.data(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        PPLCUDAConvolutionCvtFlt(stream, weight_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                                 shape_in0.GetDataType(), temp_conv_param);

        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        options.tensors->find(preedge_id)->second->GetShape() = postshape;
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }
    reinterpret_cast<CudaConvParam*>(options.param)->extra_param.algo_info.is_initializer_weight =
        weight_iter != data->constants.end();
    if (attr_param_.param.bias_term == 0) {
        return RC_SUCCESS;
    }

    // Split bias format to group padding
    auto bias_edge = topo->GetEdgeById(node->GetInput(2));
    auto bias_node = topo->GetNodeById(bias_edge->GetProducer());
    auto bias_iter = data->constants.find(bias_node->GetInput(0));
    if (bias_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(bias_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = bias_node->GetInput(0);
        auto postedge_id = node->GetInput(2);
        auto preshape = options.tensors->find(preedge_id)->second->GetShape();
        auto postshape = options.tensors->find(postedge_id)->second->GetShape();
        auto newshape = postshape;
        newshape.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);
        RuntimeConstantInfo bias_constat_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            bias_constat_info.Reshape(postshape); // give the init shape, but the actual shape is padded
            bias_constat_info.SetBuffer(buffer, options.device, true);
        }

        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, postshape.GetBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = options.device->GetDataConverter()->ConvertFromHost(&temp_buffer, postshape,
                                                                     bias_iter->second.data.data(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        PPLCUDAConvolutionCvtBias(stream, bias_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                                  shape_in0.GetDataType(), temp_conv_param);
        options.info->constants.emplace(preedge_id, std::move(bias_constat_info));
        options.tensors->find(preedge_id)->second->GetShape() = postshape;
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }
    return RC_SUCCESS;
}

void TuringHMMAImpgemm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                       dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = &tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1)
            shape->SetDataFormat(input_format);
        else
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = &tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
