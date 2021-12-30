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

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"

namespace PPLCUDA {

inline int AlignDataFormat(ppl::common::dataformat_t dt)
{
    switch (dt) {
        case ppl::common::DATAFORMAT_N2CX:
            return 2;
        case ppl::common::DATAFORMAT_N4CX:
            return 4;
        case ppl::common::DATAFORMAT_N8CX:
            return 8;
        case ppl::common::DATAFORMAT_N16CX:
            return 16;
        case ppl::common::DATAFORMAT_N32CX:
            return 32;
        case ppl::common::DATAFORMAT_NHWC8:
            return 8;
        case ppl::common::DATAFORMAT_NHWC16:
            return 16;
        default:
            return 1;
    }
}

} // namespace PPLCUDA

struct ReFormatParam {
    int64_t n_inner;
    int64_t n_outer;
    int64_t channel;

    int64_t src_pad;
    int64_t dst_pad;

    int64_t out_elems;
    int64_t in_elems;

    int64_t in_group;
    int64_t out_group;

    bool mix_type;
    bool mix_format;

    bool same_scale = 1;
    int quant_stride = 1;
    int quant_dim_size = 1;//output channel size
    int i_zero_point = 0;
    int o_zero_point = 0;
    float *i_step_ptr = nullptr;
    float *o_step_ptr = nullptr;
    
    bool per_channel = false;

    float i_step = 1.0f;
    float o_step = 1.0f;

    ppl::common::dataformat_t in_format;
    ppl::common::dataformat_t out_format;

    ppl::common::datatype_t in_type;
    ppl::common::datatype_t out_type;
};

enum CVTFormatMode {
    CVTFormatUnknown = 0,

    NDARRAY_N4CX = 2,
    N4CX_NDARRAY = 11,

    NDARRAY_NHWC  = 31,
    NHWC_NDARRAY  = 32,
    NHWC8_NHWC16  = 33,
    NHWC16_NHWC8  = 34,
};

enum CVTTypeMode {
    CVTTypeUnknown  = 0,
    INT8_FLOAT32    = 1,
    FLOAT32_INT8    = 2,
    FLOAT32_INT4B   = 3,
    INT4B_FLOAT32   = 4,
    INT8_FLOAT16    = 5,
    FLOAT16_INT8    = 6,
    FLOAT32_FLOAT16 = 7,
    FLOAT16_FLOAT32 = 8,
    INT4B_FLOAT16   = 9,
    FLOAT16_INT4B   = 10,
    INT8_INT4B      = 11,
    INT4B_INT8      = 12,
    INT8_INT8       = 13,
    INT4B_INT4B     = 14,
    INT32_INT64     = 15,
    INT64_INT32     = 16,
    FLOAT32_INT64   = 17,
    INT64_FLOAT32   = 18,
};

bool IsFloatEqual(const std::vector<float>& a, const std::vector<float>& b);
bool EqualQuant(const ppl::nn::cuda::CudaTensorQuant& quant_a, const ppl::nn::cuda::CudaTensorQuant& quant_b);

CVTFormatMode GetCVTFormatMode(ReFormatParam param);
CVTTypeMode GetCVTTypeMode(ReFormatParam param);

void PPLCUDACVTFormat(cudaStream_t stream, const void* input, void* output, ReFormatParam param);
void PPLCUDACVTTypePerTensor(cudaStream_t stream, const void* input, void* output, ReFormatParam param);
void PPLCUDACVTTypePerChannel(cudaStream_t stream, const void* input, void* output, ReFormatParam param);

ppl::common::RetCode SetReLayoutParam(ReFormatParam* param, const ppl::nn::TensorShape& input, const ppl::nn::TensorShape& output);
ppl::common::RetCode SetReLayoutParam(ReFormatParam* param, const ppl::nn::TensorShape& input, const ppl::nn::cuda::CudaTensorQuant& input_quant, const ppl::nn::TensorShape& output, const ppl::nn::cuda::CudaTensorQuant& output_quant);

void PPLCUDADataConvert(cudaStream_t stream, const void* input, void* output, void* tempBuf, ReFormatParam& param);
