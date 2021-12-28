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

void DepthwiseDirect::DeleteAttrParam(void*& param) {
    delete (CudaConvParam*)param;
    return;
}

void DepthwiseDirect::GetAttrParam(void*& param) const {
    if (param == nullptr) {
        param = new CudaConvParam();
    }
    *(CudaConvParam*)param = attr_param_;
    return;
}

bool DepthwiseDirect::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                  dataformat_t input_format) const {
    uint32_t group = (reinterpret_cast<CudaConvParam*>(options.param))->param.group;
    // check if conv is depthwise
    auto tensor1 = options.tensors->find(node->GetInput(1))->second->GetShape();
    if (group != tensor1.GetDim(0) || tensor1.GetDim(1) != 1 || group == 1) {
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

double DepthwiseDirect::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "DepthwiseDirect";
    attr_param_.extra_param.algo_info.kid = 0;

    // If the node has selcted, return answer directly
    auto pair = selection_res_.find(node->GetId());
    if (pair != selection_res_.end()) {
        attr_param_.extra_param.algo_info.kid = pair->second.kernel_index;
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

    auto input_id0 = options.tensors->find(node->GetInput(0))->second->GetEdge()->GetId();
    auto input_id1 = options.tensors->find(node->GetInput(1))->second->GetEdge()->GetId();
    auto input_quant0 = options.quants->at(input_id0);
    auto input_quant1 = options.quants->at(input_id1);
    auto output_id = options.tensors->find(node->GetOutput(0))->second->GetEdge()->GetId();
    auto output_quant = options.quants->at(output_id);

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
    ALLOC_BUFFERF_FOR_ALGO_SELECT(quant_buffer, shape_in1.GetDim(0) * sizeof(float), ALGO_MAX_TIME)

    // Do select
    auto stream = options.device->GetStream();
    auto kernel_id = PPLCUDADepthwiseSelectKernel(stream, input_buffer.addr, weight_buffer.addr, bias_buffer.addr, 1,
                                                  temp_conv_param, temp_fuse_param, output_buffer.addr, shape_out.GetDataType(), input_quant0.scale[0], (float*)quant_buffer.addr, output_quant.scale[0]);
    attr_param_.extra_param.algo_info.kid = kernel_id;

    auto run_begin_ts = std::chrono::system_clock::now();
    PPLCUDADepthwiseForwardCudaImp(stream, kernel_id, input_buffer.addr, weight_buffer.addr, bias_buffer.addr,
                                   temp_conv_param, temp_fuse_param, output_buffer.addr, shape_out.GetDataType(), input_quant0.scale[0], (float*)quant_buffer.addr, output_quant.scale[0]);
    auto run_end_ts = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    double timer = (double)diff.count() / 1000;

    LOG(DEBUG) << "Select DepthwiseDirect algorithm with kernel index " << attr_param_.extra_param.algo_info.kid
               << " and excute timer " << timer << " for node[" << node->GetName() << "]";

    SelectionInfo temp_res(kernel_id, 1, 1, timer);
    selection_res_.emplace(node->GetId(), std::move(temp_res));
    return timer;
}

RetCode DepthwiseDirect::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));

    auto shape_in0 = options.tensors->find(node->GetInput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdgeById(node->GetInput(1));
    auto weight_node = topo->GetNodeById(weight_edge->GetProducer());
    auto preedge_id = weight_node->GetInput(0);
    auto postedge_id = node->GetInput(1);

    auto weight_iter = data->constants.find(preedge_id);
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(preedge_id) == options.info->constants.end()) {
        auto preshape = options.tensors->find(preedge_id)->second->GetShape();
        auto postshape = options.tensors->find(postedge_id)->second->GetShape();
        auto newshape = postshape;
        newshape.SetDim(0, (postshape.GetDim(0) + align_size - 1) / align_size * align_size);

        BufferDesc temp_buffer;
        auto status = options.device->Realloc(postshape, &temp_buffer);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        BufferDescGuard __device_src_guard__(&temp_buffer, [&options](BufferDesc* buffer) {
            options.device->Free(buffer);
        });

        RuntimeConstantInfo constant_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            constant_info.Reshape(postshape); // give the init shape, but the actual shape is padded
            constant_info.SetBuffer(buffer, options.device, true);
        }

        status = options.device->GetDataConverter()->ConvertFromHost(&temp_buffer, postshape,
                                                                     weight_iter->second.data.data(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        conv_param_t temp_conv_param;
        auto shape_in0 = options.tensors->find(node->GetInput(0))->second->GetShape();
        auto shape_in1 = options.tensors->find(node->GetInput(1))->second->GetShape();
        auto shape_out = options.tensors->find(node->GetOutput(0))->second->GetShape();


        ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_.param, temp_conv_param);
        auto stream = options.device->GetStream();
        PPLCUDADepthwiseConvertFilter(stream, temp_buffer.addr, constant_info.GetBufferDesc().addr, temp_conv_param, shape_out.GetDataType());

        options.info->constants.emplace(preedge_id, std::move(constant_info));
        options.tensors->find(preedge_id)->second->GetShape() = postshape;
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }

    reinterpret_cast<CudaConvParam*>(options.param)->extra_param.algo_info.is_initializer_weight =
        weight_iter != data->constants.end();
    return RC_SUCCESS;
}

void DepthwiseDirect::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and fused nodes
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID)
            continue;
        auto shape = &tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1 && i != 1)
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
