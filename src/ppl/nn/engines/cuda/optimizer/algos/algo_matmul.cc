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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_matmul.h"

#include <chrono>

#include "ppl/common/cuda/cuda_types.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/bgemm.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

void MatMulAlgorithm::DeleteAttrParam(void*& param) {
    delete (CudaGemmParam*)param;
    return;
}

void MatMulAlgorithm::GetAttrParam(void*& param) const {
    if (param == nullptr) {
        param = new CudaGemmParam();
    }
    *(CudaGemmParam*)param = attr_param_;
    return;
}

bool MatMulAlgorithm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                  dataformat_t input_format) const {
    // check if matmul is fp32 type
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    if (tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        return false;
    }
    // check if matmul is quantization
    auto quant0 = options.quants->at(node->GetInput(0));
    if (quant0.type == DATATYPE_INT8 && input_format != DATAFORMAT_NHWC16) {
        return false;
    }
    if (quant0.type == DATATYPE_FLOAT16 && input_format != DATAFORMAT_NHWC8) {
        return false;
    }
    if (quant0.type == DATATYPE_FLOAT32) {
        return false;
    }
    return true;
}

double MatMulAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaGemmParam*>(options.param));
    options.compile_set->emplace(node->GetId());
    if (node->GetInputCount() == 3) {
        attr_param_.extra_param.bias_term = true;
    }

    auto shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_in2 = TensorShape();
    auto shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    auto dim_count0 = shape_in0.GetDimCount();
    auto dim_count1 = shape_in1.GetDimCount();
    auto out_dim_count = shape_out.GetDimCount();

    { // Give the default kernel
        if (shape_in0.GetDataType() == DATATYPE_FLOAT16) {
            attr_param_.extra_param.algo_info.algo_name = "nv2spkSm75Fp16Conv_hmma1688_nhwc_f1_b128x128_w64x64_k32_s32_buf1";
        } else if (shape_in0.GetDataType() == DATATYPE_INT8) {
            attr_param_.extra_param.algo_info.algo_name = "nv2spkSm75Int8Conv_imma8816_nhwc_f1_b64x64_w64x32_k32_s16_buf1";
        } else {
            return ALGO_MAX_TIME;
        }
        attr_param_.extra_param.algo_info.kid = 0; // TODO
        attr_param_.extra_param.algo_info.splitk = 1;
        attr_param_.extra_param.algo_info.splitf = 1;
        attr_param_.extra_param.algo_info.ParseAlgoName();
    }

    if (dim_count0 < 2 || dim_count1 < 2) {
        return 0.0f;
    }

    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    temp_conv_param.in_num =
        attr_param_.param.transA ? shape_in0.GetDim(dim_count0 - 1) : shape_in0.GetDim(dim_count0 - 2);
    int m_id = dim_count0 - 2;
    if (temp_conv_param.in_num == 1) {
        int m_id = dim_count0 - 3;
        while (m_id && shape_in0.GetDim(m_id) == 1)
            m_id--;
        temp_conv_param.in_num = shape_in0.GetDim(m_id);
    }
    int batch = 1;
    for (int i = 0; i < m_id; i++) {
        batch *= shape_in0.GetDim(i);
    }
    if (dim_count1 == 2) {
        temp_conv_param.in_num *= batch;
        batch = 1;
    }
    temp_conv_param.num_chl =
        attr_param_.param.transB ? shape_in1.GetDim(dim_count1 - 1) : shape_in1.GetDim(dim_count1 - 2);
    temp_conv_param.num_flt =
        attr_param_.param.transB ? shape_in1.GetDim(dim_count1 - 2) : shape_in1.GetDim(dim_count1 - 1);
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
    temp_conv_param.has_bias = 0;

    const std::string& key_str = node->GetName();
    auto algo_info = options.algos->find(key_str);
    if (algo_info != options.algos->end()) {
        attr_param_.extra_param.algo_info.kid = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
	    attr_param_.extra_param.algo_info.gemm_batch = batch;
        attr_param_.extra_param.algo_info.algo_name = algo_info->second.kname;
        if (algo_info->second.splitk > 1)
            attr_param_.extra_param.algo_info.algo_name += "_spk" + std::to_string(algo_info->second.splitk);
        attr_param_.extra_param.algo_info.ParseAlgoName();
        return 0.0f;
    }

    if (options.args->quick_select) {
        return 0.0f;
    }

    // Padding
    auto K = shape_in0.GetDim(dim_count0 - 1);
    auto N = shape_in1.GetDim(dim_count1 - 1);

    shape_in0.SetDim(dim_count0 - 1, (K + align_size - 1) / align_size * align_size);
    shape_in1.SetDim(dim_count1 - 2, (K + align_size - 1) / align_size * align_size);
    shape_out.SetDim(out_dim_count - 1, (N + align_size - 1) / align_size * align_size);
    if (attr_param_.extra_param.bias_term) {
        shape_in2 = *options.tensors->find(node->GetInput(2))->second->GetShape();
        shape_in2.SetDim(0, (shape_in2.GetDim(0) + align_size - 1) / align_size * align_size);
    }

    RetCode status;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(input_buffer, shape_in0.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(weight_buffer, shape_in1.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(bias_buffer, shape_in2.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(output_buffer, shape_out.CalcBytesIncludingPadding(), ALGO_MAX_TIME)

    uint64_t size = PPLBgemmCUDAGetBufSize(&shape_in0, attr_param_.param.transA);
    ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, size, ALGO_MAX_TIME)

    auto stream = options.device->GetStream();

    double timer = ALGO_MAX_TIME;
    int device_id = options.device->GetDeviceId();
#ifdef PPLNN_ENABLE_CUDA_JIT
    // Do select
    LOG(INFO) << "Compiling " << node->GetName();
    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        timer = PPLCUDABgemmJITSelectKernel(device_id, stream, shape_in0.GetDataType(), &shape_in0, input_buffer.addr,
                                            &shape_in1, weight_buffer.addr, bias_buffer.addr, &shape_out,
                                            output_buffer.addr, temp_buffer.addr, temp_conv_param, temp_fuse_param,
                                            attr_param_.extra_param.algo_info);
    }
    LOG(INFO) << "select kernel " << attr_param_.extra_param.algo_info.algo_name;
#else
    // Do Select
    if (shape_in0.GetDataType()==ppl::common::DATATYPE_FLOAT16) {
        timer = PPLCUDABgemmSelectKernel(device_id, stream, &shape_in0, input_buffer.addr, &shape_in1, weight_buffer.addr,
                                         &shape_out, output_buffer.addr, temp_buffer.addr,
                                         attr_param_.param, temp_fuse_param, attr_param_.extra_param.algo_info);
    }
#endif
    CudaArgs::AlgoSelects algo_select;
    algo_select.kname = attr_param_.extra_param.algo_info.algo_name;
    algo_select.kid = attr_param_.extra_param.algo_info.kid;
    algo_select.splitk = attr_param_.extra_param.algo_info.splitk;
    algo_select.splitf = attr_param_.extra_param.algo_info.splitf;
    options.algos->emplace(key_str, std::move(algo_select));
    LoadAlgoInfo(options.args->save_algo_path, attr_param_.extra_param.algo_info, key_str);
    return timer;
}

RetCode MatMulAlgorithm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    return RC_SUCCESS;
}

void MatMulAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1) {
            shape->SetDataFormat(input_format);
        } else {
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
