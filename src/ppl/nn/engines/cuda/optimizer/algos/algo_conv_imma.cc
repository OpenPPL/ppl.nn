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

void TuringIMMAImpgemm::DeleteAttrParam(void*& param) {
    delete (CudaConvParam*)param;
    return;
}

void TuringIMMAImpgemm::GetAttrParam(void*& param) const {
    if (param == nullptr)
        param = new CudaConvParam();
    *(CudaConvParam*)param = attr_param_;
    return;
}

bool TuringIMMAImpgemm::IsSupported(const ir::Node* node, const OptKernelOptions& options) const {
    // check if conv quant to INT8
    auto quant0 = options.quants->at(node->GetInput(0));
    if (quant0.type != DATATYPE_INT8) {
        return false;
    }
    return true;
}

double TuringIMMAImpgemm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "TuringIMMAImpgemm";
    attr_param_.extra_param.algo_info.kernel_index = 0;
    double timer = 1e-4f; // TODO: add SelectAlgo
    return timer;
}

RetCode TuringIMMAImpgemm::ModifyParam(const ir::Node* node, OptKernelOptions& options) {
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

void TuringIMMAImpgemm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
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
