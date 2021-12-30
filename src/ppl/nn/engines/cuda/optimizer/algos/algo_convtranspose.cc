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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_convtranspose.h"

#include <chrono>

#include "ppl/common/cuda/cuda_types.h"
#include "cudakernel/gemm/gemm.h"
#include "cudakernel/nn/convtranspose.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"

//#include "cudakernel/gemm/gemm.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

void ConvTransposeAlgorithm::DeleteAttrParam(void*& param) {
    delete (CudaConvTransposeParam*)param;
    return;
}

void ConvTransposeAlgorithm::GetAttrParam(void*& param) const {
    if (param == nullptr) {
        param = new CudaConvTransposeParam();
    }
    *(CudaConvTransposeParam*)param = attr_param_;
    return;
}

double ConvTransposeAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvTransposeParam*>(options.param));
    options.compile_set->emplace(node->GetId());

    auto shape_in0 = options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_in2 = TensorShape();
    auto shape_out = options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    temp_conv_param.in_num = shape_in0.GetDim(0);
    temp_conv_param.num_chl = shape_in1.GetDim(0);
    temp_conv_param.num_flt = shape_in1.GetDim(1);
    temp_conv_param.in_height = 1;
    temp_conv_param.in_width = 1;
    temp_conv_param.flt_height = 1;
    temp_conv_param.flt_width = 1;
    temp_conv_param.out_height = 1;
    temp_conv_param.out_width = 1;
    temp_conv_param.pad_height = 1;
    temp_conv_param.pad_width = 1;
    temp_conv_param.stride_height = 1;
    temp_conv_param.stride_width = 1;
    temp_conv_param.hole_height = 1;
    temp_conv_param.hole_width = 1;
    temp_conv_param.num_grp = 1;
    temp_conv_param.has_bias = node->GetInputCount() > 2;

    std::string key_str = node->GetName();
    auto algo_info = options.algos->find(key_str);
    if (algo_info != options.algos->end()) {
        attr_param_.extra_param.algo_info.algo_name = algo_info->second.kname;
        attr_param_.extra_param.algo_info.kid = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
        PPLCUDAConvolutionLoadAlgoParam(attr_param_.extra_param.algo_info);
        return 0.0f;
    }  else { // Give the default kernel
        attr_param_.extra_param.algo_info.algo_name = "nv2spkConv_hmma1688_nhwc_f1_b128x128_w64x64_k32_s32_buf1";
        attr_param_.extra_param.algo_info.kid = 0;
        attr_param_.extra_param.algo_info.splitk = 1;
        attr_param_.extra_param.algo_info.splitf = 1;
        PPLCUDAConvolutionLoadAlgoParam(attr_param_.extra_param.algo_info);
    }

    if (options.args->quick_select) {
        return 0.0f;
    }

    // Padding
    shape_in1.SetDim(0, (shape_in1.GetDim(0) + align_size - 1) / align_size * align_size);
    shape_in1.SetDim(1, (shape_in1.GetDim(1) + align_size - 1) / align_size * align_size);
    if (temp_conv_param.has_bias) {
        shape_in2 = options.tensors->find(node->GetInput(2))->second->GetShape();
        shape_in2.SetDim(0, (shape_in2.GetDim(0) + align_size - 1) / align_size * align_size);
    }

    RetCode status;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(input_buffer, shape_in0.GetBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(weight_buffer, shape_in1.GetBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(bias_buffer, shape_in2.GetBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(output_buffer, shape_out.GetBytesIncludingPadding(), ALGO_MAX_TIME)

    uint64_t size = PPLGemmCUDAGetBufSize(&shape_in0, 0);
    ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, size, ALGO_MAX_TIME)

    auto stream = options.device->GetStream();

#ifdef PPLNN_ENABLE_CUDA_JIT
    // Do select
    LOG(INFO) << "Compiling " << node->GetName();
    int device_id = options.device->GetDeviceId();
    PPLCUDAConvolutionPredictKernel(shape_in0.GetDataType(), attr_param_.extra_param.algo_info, temp_conv_param);
    auto timer = PPLCUDAGemmJITSelectKernel(device_id, stream, shape_in0.GetDataType(), &shape_in0, input_buffer.addr, &shape_in1,
                                            weight_buffer.addr, bias_buffer.addr, &shape_out, output_buffer.addr,
                                            temp_buffer.addr, temp_conv_param, temp_fuse_param,
                                            attr_param_.extra_param.algo_info);
#else
    // Do Select
    ppl::nn::common::GemmParam param;
    param.transA = 0;
    param.transB = 0;
    param.bias_term = temp_conv_param.has_bias;
    auto timer = PPLCUDAGemmSelectKernel(stream, &shape_in0, input_buffer.addr, &shape_in1, weight_buffer.addr,
                                         bias_buffer.addr, &shape_out, output_buffer.addr, temp_buffer.addr,
                                         param, temp_fuse_param, attr_param_.extra_param.algo_info);
#endif
    CudaArgs::AlgoSelects algo_select;
    algo_select.kname  = attr_param_.extra_param.algo_info.algo_name;
    algo_select.kid    = attr_param_.extra_param.algo_info.kid;
    algo_select.splitk = attr_param_.extra_param.algo_info.splitk;
    algo_select.splitf = attr_param_.extra_param.algo_info.splitf;
    options.algos->emplace(key_str, std::move(algo_select));
    LoadAlgoInfo(options.args->save_algo_path, attr_param_.extra_param.algo_info, key_str);
    return timer;
}

RetCode ConvTransposeAlgorithm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvTransposeParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdgeById(node->GetInput(1));
    auto weight_node = topo->GetNodeById(weight_edge->GetProducer());

    auto shape_in0 = options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_out = options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    RetCode status;
    auto stream = options.device->GetStream();
    auto weight_iter = data->constants.find(weight_node->GetInput(0));
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(weight_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = weight_node->GetInput(0);
        auto postedge_id = node->GetInput(1);
        auto preshape = options.tensors->find(preedge_id)->second->GetShape();
        auto postshape = options.tensors->find(postedge_id)->second->GetShape();
        auto newshape = postshape;
        newshape.SetDim(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size);

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

        auto size = pplConvTransposeGetFilterBufSizeCudaFp32(shape_in1.GetDim(0), shape_in1.GetDim(1), shape_in1.GetDim(2), shape_in1.GetDim(3));
        ALLOC_BUFFERF_FOR_ALGO_SELECT(filter_temp_buffer, size, RC_OUT_OF_MEMORY)
        ALLOC_BUFFERF_FOR_ALGO_SELECT(filter_input_buffer, postshape.GetBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = options.device->GetDataConverter()->ConvertFromHost(&filter_input_buffer, postshape,
                                                                     weight_iter->second.data.data(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        PPLCUDAConvTransposeCvt(stream, filter_input_buffer.addr, filter_temp_buffer.addr, weight_constat_info.GetBufferDesc().addr, 
                                &shape_in1, &attr_param_.param);

        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        options.tensors->find(preedge_id)->second->GetShape() = postshape;
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }
    reinterpret_cast<CudaConvTransposeParam*>(options.param)->extra_param.algo_info.is_initializer_weight =
        weight_iter != data->constants.end();
    return RC_SUCCESS;
}

void ConvTransposeAlgorithm::ReshapeOnEdges(const ir::Node* node,
                                            std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                            dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = &tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1) {
            shape->SetDataFormat(input_format);
        } else {
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = &tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
