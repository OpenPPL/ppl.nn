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

#include "cudakernel/memory/pad.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

template <typename T>
__global__ void ppl_cukernel_pad(
    int64_t num_elems,
    int num_dims,
    PadKernelParam param,
    GArray<int64_t> input_dims,
    GArray<int64_t> input_strides,
    const T* input,
    const int64_t* pads,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    bool use_pad_value   = false;
    int64_t input_offset = 0;
    int out_idx, remain = index;
    for (int it = 0; (it < num_dims) && !use_pad_value; ++it) {
        output_strides_fast[it].divmod(remain, out_idx, remain);
        int64_t start_pad_val = pads[it];
        int in_idx            = 0;
        if (out_idx < start_pad_val) {
            switch (param.mode) {
                case 0: // constant
                    use_pad_value = true;
                    break;
                case 1: // reflect
                    in_idx = start_pad_val - out_idx;
                    break;
                case 2: // edge
                    in_idx = 0;
                    break;
            }
        } else if (out_idx >= start_pad_val + input_dims[it]) {
            switch (param.mode) {
                case 0: // constant
                    use_pad_value = true;
                    break;
                case 1: // reflect
                    in_idx = input_dims[it] - 2 -
                             (out_idx - (start_pad_val + input_dims[it]));
                    break;
                case 2: // edge
                    in_idx = input_dims[it] - 1;
                    break;
            }
        } else {
            in_idx = out_idx - start_pad_val;
        }
        input_offset += in_idx * input_strides[it];
    }
    output[index] = use_pad_value ? (T)param.constant_value : input[input_offset];
}

ppl::common::RetCode PPLCUDAPadForwardImp(
    cudaStream_t stream,
    PadKernelParam param,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* pads_shape,
    const int64_t* pads,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    int block_size     = 256;
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    int num_dims       = output_shape->GetDimCount();
    GArray<int64_t> input_dims(num_dims);
    GArray<int64_t> input_strides(num_dims);
    GArray<DivModFast> output_strides_fast(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_dims[it]          = input_shape->GetDim(it);
        input_strides[it]       = acc_input_stride;
        output_strides_fast[it] = DivModFast(acc_output_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }

    switch (input_shape->GetDataType()) {
        case ppl::common::DATATYPE_FLOAT16: {
            ppl_cukernel_pad<<<grid_size, block_size, 0, stream>>>(
                num_elems, num_dims, param, input_dims, input_strides, (const half*)input, pads, output_strides_fast, (half*)output);
            return ppl::common::RC_SUCCESS;
        }
        case ppl::common::DATATYPE_FLOAT32: {
            ppl_cukernel_pad<<<grid_size, block_size, 0, stream>>>(
                num_elems, num_dims, param, input_dims, input_strides, (const float*)input, pads, output_strides_fast, (float*)output);
            return ppl::common::RC_SUCCESS;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
}
