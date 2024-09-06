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

bool TuringHMMAImpgemm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                    dataformat_t input_format) const {
    uint32_t group = (reinterpret_cast<CudaConvParam*>(options.param))->param.group;
    // check if conv is depthwise
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& tensor1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    if (tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        return false;
    }
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

// copy from pplnn/src/ppl/nn/engines/cuda/params/conv_extra_param.cc
int GetRelueType_(const std::string& name) {
    if (name == "Relu" || name == "ReLU" || name == "ReLU3d")
        return 0;
    if (name == "Clip")
        return 1;
    if (name == "PRelu")
        return 2;
    if (name == "LeakyRelu")
        return 3;
    if (name == "Sigmoid")
        return 4;
    return -1;
}

// modify from pplnn/src/ppl/nn/engines/cuda/params/conv_extra_param.cc
// just use for conv algo select, not stand for the true fusion param
RetCode ConvertToForwardFuseParam_(const ConvFusionInfo& fuse_info, fuse_param_t& fuse_param) {
    const std::set<std::string> relu_set{"Relu", "ReLU", "ReLU3d", "Clip", "PRelu", "LeakyRelu", "Sigmoid"};
    int fuse_index = 0;
    int fuse_size = fuse_info.types.size();

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType_(fuse_info.types[fuse_index]);
        switch (type) {
            case 0: // Relu
                fuse_param.has_activation = 1;
                break;
            case 1: // Clip
                fuse_param.has_clip = true;
                break;
            case 2: // PRelu
                if (fuse_index == 0) fuse_param.has_prelu = 1;
                else fuse_param.has_elt_prelu = 1;
                break;
            case 3: // LeakyRelu
                if (fuse_index == 0) fuse_param.has_prelu = 1;
                else fuse_param.has_elt_prelu = 1;
                break;
            case 4: // Sigmoid
                fuse_param.has_activation = 2;
                break;
            default:
                return RC_UNSUPPORTED;
        }
        fuse_index++;
    }

    if (fuse_index < fuse_size && (fuse_info.types[fuse_index] == "Add" || fuse_info.types[fuse_index] == "Eltwise")) {
        fuse_param.has_elt = true;
        fuse_index++;
    }

    if (fuse_index < fuse_size && fuse_param.has_elt && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType_(fuse_info.types[fuse_index]);
        switch (type) {
            case 0: // Relu
                fuse_param.has_elt_activation = 1;
                break;
            case 1: // Clip
                fuse_param.has_elt_clip = true;
                break;
            case 2: // PRelu
                if (fuse_index == 0) fuse_param.has_prelu = 1;
                else fuse_param.has_elt_prelu = 1;
                break;
            case 3: // LeakyRelu
                if (fuse_index == 0) fuse_param.has_prelu = 1;
                else fuse_param.has_elt_prelu = 1;
                break;
            case 4: // Sigmoid
                fuse_param.has_elt_activation = 2;
                break;
            default:
                return RC_UNSUPPORTED;
        }
    }

    fuse_param.has_concat = fuse_info.channel_offset >= 0;

    return RC_SUCCESS;
}

double TuringHMMAImpgemm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "TuringHMMAImpgemm";
    options.compile_set->emplace(node->GetId());

    const std::string& key_str = node->GetName();
    auto algo_info = options.args->alog_selects.find(key_str);
    if (algo_info != options.args->alog_selects.end()) {
        attr_param_.extra_param.algo_info.kid = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
        attr_param_.extra_param.algo_info.algo_type = algo_info->second.ktype;
        attr_param_.extra_param.algo_info.algo_name = algo_info->second.kname;
        if (algo_info->second.splitk > 1)
            attr_param_.extra_param.algo_info.algo_name += "_spk" + std::to_string(algo_info->second.splitk);
        attr_param_.extra_param.algo_info.ParseAlgoName();
        return 0.0f;
    } else { // Give the default kernel
#ifdef PPLNN_CUDA_ENABLE_KERNEL_CUT
        attr_param_.extra_param.algo_info.algo_name = "nvSwzlSm75Fp16Conv_hmma1688_nhwc_fn_b32x256_w32x64_k8_buf2";
        attr_param_.extra_param.algo_info.kid = 685;
#else
        attr_param_.extra_param.algo_info.algo_name = "nvSwzlSm75Fp16Conv_hmma1688_nhwc_fn_b128x64_w64x32_k64_buf2";
        attr_param_.extra_param.algo_info.kid = 5197;
#endif
        attr_param_.extra_param.algo_info.splitk = 1;
        attr_param_.extra_param.algo_info.splitf = 1;
        attr_param_.extra_param.algo_info.ParseAlgoName();
    }

    auto shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_in2 = TensorShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_, temp_conv_param);
    ConvertToForwardFuseParam_(attr_param_.extra_param.fuse_info, temp_fuse_param);     // will reset to default during kernel selection

    // input shape is invalid
    if (shape_in0.GetDimCount() != 4 || shape_in1.GetDimCount() != 4) {
        return 0.0f;
    }
    // input H or W is too small
    if (shape_in0.GetDim(2) + 2 * temp_conv_param.pad_height < shape_in1.GetDim(2) ||
        shape_in0.GetDim(3) + 2 * temp_conv_param.pad_width < shape_in1.GetDim(3)) {
        shape_in0.SetDim(2, shape_in1.GetDim(2));
        shape_in0.SetDim(3, shape_in1.GetDim(3));
    }

    if (options.args->quick_select) {
        return 0.0f;
    }

    // Padding
    shape_in0.SetDim(1, shape_in1.GetDim(1) * attr_param_.param.group);
    uint32_t k_per_grp = shape_in1.GetDim(0) / attr_param_.param.group;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;
    shape_in1.SetDim(0, k_per_grp_pad * attr_param_.param.group);
    if (temp_conv_param.has_bias) {
        shape_in2 = *options.tensors->find(node->GetInput(2))->second->GetShape();
        shape_in2.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);
    }

    RetCode status;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(input_buffer, shape_in0.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(weight_buffer, shape_in1.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(bias_buffer, shape_in2.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(output_buffer, shape_out.CalcBytesIncludingPadding(), ALGO_MAX_TIME)

    uint64_t size = PPLCUDAConvolutionGetCompilationBufSize(shape_in0.GetDataType(), temp_conv_param);
    ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, size, ALGO_MAX_TIME)

    auto stream = options.device->GetStream();

#ifdef PPLNN_ENABLE_CUDA_JIT
    // Do select
    LOG(INFO) << "Compiling " << node->GetName();
    auto timer = PPLCUDAConvolutionJitSelectKernel(
        options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)input_buffer.addr,
        (int4*)weight_buffer.addr, (int4*)output_buffer.addr, (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
        attr_param_.extra_param.algo_info, temp_conv_param, temp_fuse_param);
    LOG(INFO) << "select kernel " << attr_param_.extra_param.algo_info.algo_name;
#else
    // Do select
    auto timer = PPLCUDAConvolutionSelectKernel(
        options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)input_buffer.addr,
        (int4*)weight_buffer.addr, (int4*)output_buffer.addr, (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
        attr_param_.extra_param.algo_info, temp_conv_param, temp_fuse_param, size);
#endif
    CudaArgs::AlgoSelects algo_select;
    algo_select.ktype = attr_param_.extra_param.algo_info.algo_type;
    algo_select.kname = attr_param_.extra_param.algo_info.algo_name;
    algo_select.kid = attr_param_.extra_param.algo_info.kid;
    algo_select.splitk = attr_param_.extra_param.algo_info.splitk;
    algo_select.splitf = attr_param_.extra_param.algo_info.splitf;
    options.args->alog_selects.emplace(key_str, std::move(algo_select));
    return timer;
}

RetCode TuringHMMAImpgemm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdge(node->GetInput(1));
    auto weight_node = topo->GetNode(weight_edge->GetProducer());

    const TensorShape& shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    RetCode status;
    conv_param_t temp_conv_param;
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_, temp_conv_param);

    uint32_t k_per_grp = shape_in1.GetDim(0) / temp_conv_param.num_grp;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;

    // Split weight format to group padding
    auto stream = options.device->GetStream();
    auto weight_iter = data->constants.find(weight_node->GetInput(0));
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(weight_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = weight_node->GetInput(0);
        auto postedge_id = node->GetInput(1);
        const TensorShape& preshape = *options.tensors->find(preedge_id)->second->GetShape();
        const TensorShape& postshape = *options.tensors->find(postedge_id)->second->GetShape();
        auto newshape = postshape;
        newshape.SetPadding1(0, k_per_grp_pad * temp_conv_param.num_grp - newshape.GetDim(0));

        RuntimeConstantInfo weight_constat_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            weight_constat_info.Reshape(newshape); // give the converted shape, but the dims are not changed
            weight_constat_info.SetBuffer(buffer, options.device, true);
        }

        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, newshape.CalcBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = options.device->ConvertFromHost(&temp_buffer, postshape, weight_iter->second.data.GetData(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        PPLCUDAConvolutionCvtFlt(stream, weight_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                                 shape_in0.GetDataType(), temp_conv_param);

        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        // record edgeid --> name mapping
        options.refit_info->name2edgeid[topo->GetEdge(preedge_id)->GetName()] = preedge_id;
        options.refit_info->edge2node[preedge_id] = node->GetType().name;

        *options.tensors->find(preedge_id)->second->GetShape() = newshape;
        *options.tensors->find(postedge_id)->second->GetShape() = newshape;
        options.quants->at(preedge_id) = options.quants->at(postedge_id);
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }
    reinterpret_cast<CudaConvParam*>(options.param)->extra_param.is_initializer_weight =
        weight_iter != data->constants.end();
    if (attr_param_.extra_param.bias_term == 0) {
        return RC_SUCCESS;
    }

    // Split bias format to group padding
    auto bias_edge = topo->GetEdge(node->GetInput(2));
    auto bias_node = topo->GetNode(bias_edge->GetProducer());
    auto bias_iter = data->constants.find(bias_node->GetInput(0));
    if (bias_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(bias_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = bias_node->GetInput(0);
        auto postedge_id = node->GetInput(2);
        const TensorShape& preshape = *options.tensors->find(preedge_id)->second->GetShape();
        const TensorShape& postshape = *options.tensors->find(postedge_id)->second->GetShape();
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

            bias_constat_info.Reshape(newshape);
            bias_constat_info.SetBuffer(buffer, options.device, true);
        }

        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, newshape.CalcBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = options.device->ConvertFromHost(&temp_buffer, postshape, bias_iter->second.data.GetData(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        PPLCUDAConvolutionCvtBias(stream, bias_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                                  shape_in0.GetDataType(), temp_conv_param);
        options.info->constants.emplace(preedge_id, std::move(bias_constat_info));
        // record edgeid --> name mapping
        options.refit_info->name2edgeid[topo->GetEdge(preedge_id)->GetName()] = preedge_id;
        options.refit_info->edge2node[preedge_id] = node->GetType().name;

        *options.tensors->find(preedge_id)->second->GetShape() = newshape;
        *options.tensors->find(postedge_id)->second->GetShape() = newshape;
        options.quants->at(preedge_id) = options.quants->at(postedge_id);
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
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1)
            shape->SetDataFormat(input_format);
        else
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
