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

#include "cudakernel/unary/clip.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename T>
__device__ __inline__ T ppl_scalar_clip(const T& in_val, float _min, float _max)
{
    float resf = (float)in_val;
    resf       = (resf > _min) ? resf : _min;
    resf       = (resf < _max) ? resf : _max;
    return (T)resf;
}

#endif

template <typename srcT, typename CalT, int Iter>
__global__ void ppl_cukernel_clip_ndarray(
    const uint64_t num_elems,
    const srcT* input,
    srcT* output,
    float _min,
    float _max)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    srcT in_val = input[index];
    srcT out_val;
    CalT* out_val_ptr = reinterpret_cast<CalT*>(&out_val);
    CalT* in_val_ptr = reinterpret_cast<CalT*>(&in_val);
    for (int it = 0; it < Iter; it++) {
        out_val_ptr[it] = ppl_scalar_clip<CalT>(in_val_ptr[it], _min, _max);
    }
    output[index] = out_val;
#endif
}

template <typename srcT, typename CalT, int Iter>
__global__ void ppl_cukernel_clip_nhwc(
    const uint64_t num_elems,
    int channels,
    int pad_channels,
    int hwc,
    int hw,
    const srcT* input,
    srcT* output,
    float _min,
    float _max)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int hwc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hwc_idx >= hwc)
        return;
    int c_idx     = hwc_idx % channels;
    int hw_idx    = hwc_idx / channels;
    int b_idx     = blockIdx.z;
    int64_t index = (b_idx * hw + hw_idx) * pad_channels + c_idx;

    srcT in_val = input[index];
    srcT out_val;
    CalT* out_val_ptr = reinterpret_cast<CalT*>(&out_val);
    CalT* in_val_ptr = reinterpret_cast<CalT*>(&in_val);
    for (int it = 0; it < Iter; it++) {
        out_val_ptr[it] = ppl_scalar_clip<CalT>(in_val_ptr[it], _min, _max);
    }
    output[index] = out_val;
#endif
}
template <typename srcT, int Iter>
__global__ void ppl_cukernel_clip_ndarray_int8(
    const uint64_t num_elems,
    const srcT* input,
    srcT* output,
    float _min,
    float _max,
    float in_scale,
    float out_scale)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    srcT in_val = input[index];
    srcT out_val;
    char* out_val_ptr = reinterpret_cast<char*>(&out_val);
    char* in_val_ptr = reinterpret_cast<char*>(&in_val);
    for (int it = 0; it < Iter; it++) {
        float tmp = (float)in_val_ptr[it] * in_scale;
        float out_val = tmp < _min ? _min : tmp > _max ? _max : tmp;
        int int_val = __float2int_rn(out_val * out_scale);
        out_val_ptr[it] = int_val < -128 ? -128 : int_val > 127 ? 127 : (char)int_val;
    }
    output[index] = out_val;
}

template <typename srcT, int Iter>
__global__ void ppl_cudakernel_clip_nhwc_int8(
    const uint64_t num_elems,
    int channels,
    int pad_channels,
    int chw,
    int hw,
    const srcT* input,
    srcT* output,
    float _min,
    float _max,
    float in_scale,
    float out_scale)
{
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw) return;
    int c_idx = chw_idx % channels;
    int hw_idx = chw_idx / channels;
    int b_idx = blockIdx.z;
    int64_t index = (b_idx * hw + hw_idx) * pad_channels + c_idx;

    srcT in_val = input[index];
    srcT out_val;
    char* out_val_ptr = reinterpret_cast<char*>(&out_val);
    char* in_val_ptr = reinterpret_cast<char*>(&in_val);
    for (int it = 0; it < Iter; it++) {
        float tmp = (float)in_val_ptr[it] * in_scale;
        float out_val = tmp < _min ? _min : tmp > _max ? _max : tmp;
        int int_val = __float2int_rn(out_val * out_scale);
        out_val_ptr[it] = int_val < -128 ? -128 : int_val > 127 ? 127 : (char)int_val;
    }
    output[index] = out_val;
   
}

ppl::common::RetCode PPLCUDAClipForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    float _min,
    float _max,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int batch          = output_shape->GetDim(0);
    int channels       = output_shape->GetDim(1);
    int pad_channels   = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int height         = output_shape->GetDimCount() > 2 ? output_shape->GetDim(2) : 1;
    int width          = output_shape->GetDimCount() > 3 ? output_shape->GetDim(3) : 1;
    int block_size = 256;
    bool can_packed = (_min <= 0.f) && (0.f <= _max); // ensure f(0) = 0

    #define PACKED_EXEC(shift, srcT, CalT, Iter) \
        int grid_size  = ((num_elems >> shift) + block_size - 1) / block_size;                  \
        ppl_cukernel_clip_ndarray<srcT, CalT, Iter><<<grid_size, block_size, 0, stream>>>(      \
            num_elems >> shift, (const srcT*)input, (srcT*)output, _min, _max);

    #define PACKED_EXEC_INT8(shift, srcT, Iter) \
        int grid_size  = ((num_elems >> shift) + block_size - 1) / block_size;                  \
        ppl_cukernel_clip_ndarray_int8<srcT, Iter><<<grid_size, block_size, 0, stream>>>(      \
            num_elems >> shift, (const srcT*)input, (srcT*)output, _min, _max, in_scale, out_scale);


    if (can_packed || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            if (num_elems & 0x1) {
                PACKED_EXEC(0, float, float, 1)
            } else if (num_elems & 0x2) {
                PACKED_EXEC(1, float2, float, 2)
            } else {
                PACKED_EXEC(2, float4, float, 4)
            }
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            if (num_elems & 0x1) {
                PACKED_EXEC(0, half, half, 1)
            } else if (num_elems & 0x2) {
                PACKED_EXEC(1, float, half, 2)
            } else if (num_elems & 0x4) {
                PACKED_EXEC(2, float2, half, 4)
            } else {
                PACKED_EXEC(3, float4, half, 8)
            }
        } else {
            if (num_elems & 0x1) {
                PACKED_EXEC_INT8(0, char, 1)
            } else if (num_elems & 0x2) {
                PACKED_EXEC_INT8(1, half, 2)
            } else if (num_elems & 0x4) {
                PACKED_EXEC_INT8(2, float, 4)
            } else if (num_elems & 0x8) {
                PACKED_EXEC_INT8(3, float2, 8)
            } else {
                PACKED_EXEC_INT8(4, float4, 16)
            }

        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int block_size = 256;
        dim3 grid_size;
        grid_size.z = batch; grid_size.y = 1;

    #define NHWC_EXEC(shift, srcT, CalT, Iter) \
        int chw = (channels >> shift) * height * width;                                      \
        grid_size.x = (chw + block_size - 1) / block_size;                  \
        ppl_cukernel_clip_nhwc<srcT, CalT, Iter><<<grid_size, block_size, 0, stream>>>(      \
            num_elems >> shift, channels >> shift, pad_channels >> shift, chw, height * width,   \
            (const srcT*)input, (srcT*)output, _min, _max);

    #define NHWC_EXEC_INT8(shift, srcT, Iter) \
        int chw = (channels >> shift) * height * width;                                      \
        grid_size.x = (chw + block_size - 1) / block_size;                  \
        ppl_cudakernel_clip_nhwc_int8<srcT, Iter><<<grid_size, block_size, 0, stream>>>(      \
            num_elems >> shift, channels >> shift, pad_channels >> shift, chw, height * width,   \
            (const srcT*)input, (srcT*)output, _min, _max, in_scale, out_scale);

        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            if (channels & 0x1) {
                NHWC_EXEC(0, float, float, 1)
            } else if (channels & 0x2) {
                NHWC_EXEC(1, float2, float, 2)
            } else {
                NHWC_EXEC(2, float4, float, 4)
            }
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            if (channels & 0x1) {
                NHWC_EXEC(0, half, half, 1)
            } else if (channels & 0x2) {
                NHWC_EXEC(1, float, half, 2)
            } else if (channels & 0x4) {
                NHWC_EXEC(2, float2, half, 4)
            } else {
                NHWC_EXEC(3, float4, half, 8)
            }
        } else { // int8
            if (channels & 0x1) {
                NHWC_EXEC_INT8(0, char, 1)
            } else if (channels & 0x2) {
                NHWC_EXEC_INT8(1, half, 2)
            } else if (channels & 0x4) {
                NHWC_EXEC_INT8(2, float, 4)
            } else if (channels & 0x8) {
                NHWC_EXEC_INT8(3, float2, 8)
            } else {
                NHWC_EXEC_INT8(4, float4, 16)
            }
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}