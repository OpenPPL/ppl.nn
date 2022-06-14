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
#include "cudakernel/common/common.h"
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
bool isFastPadSupported(const std::vector<int32_t>& pads, int32_t num_dims) {
    if (num_dims < 3) return false;
    int32_t diff_cnt = num_dims - 2;
    for (int32_t i = 0; i < diff_cnt; ++i) {
        if (pads[i] != 0) return false; // start
        if (pads[num_dims + i] != 0) return false; //end
    }
    if (pads[num_dims - 1] != 0) return false;
    if (pads[num_dims - 2] != 0) return false;
    return true;
}

template <typename T>
__global__ void ppl_cukernel_pad_fast(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int b_idx = blockIdx.z;
    int dst_idx = b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt;
    if (dst_hgt >= src_height || dst_wdt >= src_width) {
        output[dst_idx] = T(0);
    } else {
        int src_idx = b_idx * src_height * src_width + dst_hgt * src_width + dst_wdt;
        output[dst_idx] = input[src_idx];
    }
}
ppl::common::RetCode PPLCUDAPadForwardImp(
    cudaStream_t stream,
    PadKernelParam param,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const int64_t* pads,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    int num_dims       = output_shape->GetDimCount();
    if (isFastPadSupported(param.pads, num_dims)) {
        int batch = input_shape->GetElementsToDimensionExcludingPadding(num_dims - 2);
        int dst_height = output_shape->GetDim(num_dims - 2);
        int dst_width  = output_shape->GetDim(num_dims - 1);
        int src_height = input_shape->GetDim(num_dims - 2);
        int src_width  = input_shape->GetDim(num_dims - 1);
        dim3 block_size(16, 16, 1);
        dim3 grid_size(DivUp(dst_width, 16), DivUp(dst_height, 16), batch);
        switch (input_shape->GetDataType()) {
            case ppl::common::DATATYPE_FLOAT16: {
                ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const half*)input, src_height, src_width, (half*)output, dst_height, dst_width);
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT32: {
                ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const float*)input, src_height, src_width, (float*)output, dst_height, dst_width);
                return ppl::common::RC_SUCCESS;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
        return ppl::common::RC_SUCCESS;
    }
    int block_size     = 256;
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
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
