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

#include "cudakernel/nn/mmcv_gridsample.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/common/tensor_shape.h"
#include <cuda_fp16.h>
#define MIN(a, b)                             (((a) < (b)) ? (a) : (b))
#define MAX(a, b)                             (((a) > (b)) ? (a) : (b))
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit - 1), MAX(in, 0))

template <typename T>
static inline __device__ T grid_sampler_unnormalize(T coord, int64_t size, bool align_corners)
{
    if (align_corners) {
        return ((coord + 1) / 2) * (size - 1);
    } else {
        return ((coord + 1) * size - 1) / 2;
    }
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename T>
static inline __device__ T reflect_coordinates(T in, int64_t twice_low, int64_t twice_high)
{
    if (twice_low == twice_high) {
        return static_cast<T>(0);
    }
    T min     = static_cast<T>(twice_low) / 2;
    T span    = static_cast<T>(twice_high - twice_low) / 2;
    in        = fabsf(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    T extra   = fmodf(in, span);
    int flips = static_cast<int>(floorf(in / span));
    if (flips % 2 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
}

// padding_mode: zeros == 0, border == 1, reflection == 2;
template <typename T>
static inline __device__ T compute_coordinates(T coord, int64_t size, int64_t padding_mode, bool align_corners)
{
    if (padding_mode == 1) {
        CLIP_COORDINATES(coord, coord, size);
    } else if (padding_mode == 2) {
        if (align_corners) {
            coord = reflect_coordinates(coord, 0, 2 * (size - 1));
        } else {
            coord = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        CLIP_COORDINATES(coord, coord, size);
    }
    return coord;
}

// Computes the pixel source index value for a grid coordinate
template <typename T>
static inline __device__ T grid_sampler_compute_source_index(T coord, int64_t size, int64_t padding_mode, bool align_corners)
{
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    coord = compute_coordinates(coord, size, padding_mode, align_corners);
    return coord;
}

static inline __device__ bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W)
{
    return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename T>
static inline T __device__ get_value_bounded(const T* data, float x, float y, int64_t W, int64_t H, int64_t sW, int64_t sH, int64_t padding_mode, bool align_corners)
{
    x = compute_coordinates(x, W, padding_mode, align_corners);
    y = compute_coordinates(y, H, padding_mode, align_corners);

    int64_t ix = static_cast<int64_t>(x);
    int64_t iy = static_cast<int64_t>(y);

    if (within_bounds_2d(iy, ix, H, W)) {
        return data[iy * sH + ix * sW];
    }
    return (T)(0);
}

template <typename T>
static inline __device__ T cubic_convolution1(T x, T A)
{
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename T>
static inline __device__ T cubic_convolution2(T x, T A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename T>
static inline __device__ void get_cubic_upsample_coefficients(T coeffs[4],
                                                              T t)
{
    T A = -0.75;

    T x1      = t;
    coeffs[0] = cubic_convolution2<T>(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1<T>(x1, A);

    // opposite coefficients
    T x2      = 1.0 - t;
    coeffs[2] = cubic_convolution1<T>(x2, A);
    coeffs[3] = cubic_convolution2<T>(x2 + 1.0, A);
}

template <typename T>
static inline __device__ T cubic_interp1d(T x0, T x1, T x2, T x3, T t)
{
    T coeffs[4];
    get_cubic_upsample_coefficients<T>(coeffs, t);

    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

// interpolation mode: bilinear == 0, nearest == 1, bicubic == 2;
__global__ void ppl_cukernel_gridsample_fp32(
    const int num,
    const int channels,
    const int height,
    const int width,
    const int in_height,
    const int in_width,
    const int num_threads,
    const float* input0,
    const float* input1,
    float* output,
    int align_corners,
    int padding_mode,
    int interpolation_mode)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_threads)
        return;
    size_t chw       = channels * height * width;
    size_t hw        = height * width;
    size_t in_hw     = in_height * in_width;
    size_t batch_idx = idx / chw;
    size_t c_idx     = (idx / hw) % channels;
    size_t h_idx     = (idx / width) % height;
    size_t w_idx     = (idx % width);

    int grid_idx = (batch_idx * hw + h_idx * width + w_idx) << 1;
    float x      = input1[grid_idx + 0];
    float y      = input1[grid_idx + 1];

    if (isinf(x) || isinf(y)) {
        output[idx] = 0;
        return;
    }

    float ix = grid_sampler_compute_source_index(x, in_width, padding_mode, align_corners);
    float iy = grid_sampler_compute_source_index(y, in_height, padding_mode, align_corners);

    // get NE, NW, SE, SW pixel values from (x, y)
    if (interpolation_mode == 0) { // bilinear
        int ix_nw = (int)floorf(ix);
        int iy_nw = (int)floorf(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        size_t in_idx = batch_idx * channels * in_hw + c_idx * in_hw;
        float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < in_width) && (iy_nw < in_height)) ? input0[in_idx + iy_nw * in_width + ix_nw] : 0;
        float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < in_width) && (iy_ne < in_height)) ? input0[in_idx + iy_ne * in_width + ix_ne] : 0;
        float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < in_width) && (iy_sw < in_height)) ? input0[in_idx + iy_sw * in_width + ix_sw] : 0;
        float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < in_width) && (iy_se < in_height)) ? input0[in_idx + iy_se * in_width + ix_se] : 0;
        float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
        output[idx]   = out_val;
    } else if (interpolation_mode == 1) { // nearest
        int64_t ix_nearest = static_cast<int64_t>(floorf(0.5 + ix));
        int64_t iy_nearest = static_cast<int64_t>(floorf(0.5 + iy));
        size_t in_idx      = batch_idx * channels * in_hw + c_idx * in_hw;
        output[idx]        = ((ix_nearest >= 0) && (iy_nearest >= 0) && (ix_nearest < in_width) && (iy_nearest < in_height)) ? input0[in_idx + iy_nearest * in_width + ix_nearest] : 0;
    } else { // bicubic
        ix = grid_sampler_unnormalize(x, in_width, align_corners);
        iy = grid_sampler_unnormalize(y, in_height, align_corners);

        float ix_nw = floorf(ix);
        float iy_nw = floorf(iy);

        const float tx = ix - ix_nw;
        const float ty = iy - iy_nw;

        const float* inp_ptr_NC = input0 + batch_idx * in_hw;
        float coefficients[4];

        // Interpolate 4 values in the x directon
        for (int64_t i = 0; i < 4; ++i) {
            coefficients[i] = cubic_interp1d<float>(
                get_value_bounded<float>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners),
                get_value_bounded<float>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners),
                get_value_bounded<float>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners),
                get_value_bounded<float>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners),
                tx);
        }

        // Interpolate in the y direction
        output[idx] = cubic_interp1d<float>(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);
    }
}

__global__ void ppl_cukernel_gridsample_fp16(
    const int num,
    const int channels,
    const int height,
    const int width,
    const int in_height,
    const int in_width,
    const int num_threads,
    const half* input0,
    const half* input1,
    half* output,
    int align_corners,
    int padding_mode,
    int interpolation_mode)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_threads)
        return;
    size_t chw       = channels * height * width;
    size_t hw        = height * width;
    size_t in_hw     = in_height * in_width;
    size_t batch_idx = idx / chw;
    size_t c_idx     = (idx / hw) % channels;
    size_t h_idx     = (idx / width) % height;
    size_t w_idx     = (idx % width);

    int grid_idx = (batch_idx * hw + h_idx * width + w_idx) << 1;
    float x      = __half2float(input1[grid_idx + 0]);
    float y      = __half2float(input1[grid_idx + 1]);

    if (isinf(x) || isinf(y)) {
        output[idx] = (half)0;
        return;
    }

    float ix = grid_sampler_compute_source_index(x, in_width, padding_mode, align_corners);
    float iy = grid_sampler_compute_source_index(y, in_height, padding_mode, align_corners);

    // get NE, NW, SE, SW pixel values from (x, y)
    if (interpolation_mode == 0) { // bilinear
        int ix_nw = (int)floorf(ix);
        int iy_nw = (int)floorf(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        size_t in_idx = batch_idx * channels * in_hw + c_idx * in_hw;
        float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < in_width) && (iy_nw < in_height)) ? __half2float(input0[in_idx + iy_nw * in_width + ix_nw]) : 0;
        float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < in_width) && (iy_ne < in_height)) ? __half2float(input0[in_idx + iy_ne * in_width + ix_ne]) : 0;
        float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < in_width) && (iy_sw < in_height)) ? __half2float(input0[in_idx + iy_sw * in_width + ix_sw]) : 0;
        float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < in_width) && (iy_se < in_height)) ? __half2float(input0[in_idx + iy_se * in_width + ix_se]) : 0;
        float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
        output[idx]   = __float2half(out_val);
    } else if (interpolation_mode == 1) { // nearest
        int64_t ix_nearest = static_cast<int64_t>(floorf(0.5 + ix));
        int64_t iy_nearest = static_cast<int64_t>(floorf(0.5 + iy));
        size_t in_idx      = batch_idx * channels * in_hw + c_idx * in_hw;
        output[idx]        = ((ix_nearest >= 0) && (iy_nearest >= 0) && (ix_nearest < in_width) && (iy_nearest < in_height)) ? input0[in_idx + iy_nearest * in_width + ix_nearest] : half(0);
    } else { // bicubic
        ix = grid_sampler_unnormalize(x, in_width, align_corners);
        iy = grid_sampler_unnormalize(y, in_height, align_corners);

        float ix_nw = floorf(ix);
        float iy_nw = floorf(iy);

        const float tx = ix - ix_nw;
        const float ty = iy - iy_nw;

        const half* inp_ptr_NC = input0 + batch_idx * in_hw;
        float coefficients[4];

        // Interpolate 4 values in the x directon
        for (int64_t i = 0; i < 4; ++i) {
            coefficients[i] = cubic_interp1d<float>(
                __half2float(get_value_bounded<half>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners)),
                __half2float(get_value_bounded<half>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners)),
                __half2float(get_value_bounded<half>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners)),
                __half2float(get_value_bounded<half>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, in_width, in_height, 1, in_width, padding_mode, align_corners)),
                tx);
        }

        // Interpolate in the y direction
        output[idx] = __float2half(cubic_interp1d<float>(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty));
    }
#endif
}

ppl::common::RetCode PPLCUDAMMCVGridSampleForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input0_shape,
    const void* input0,
    ppl::nn::TensorShape* input1_shape,
    const void* input1,
    ppl::nn::TensorShape* output_shape,
    void* output,
    ppl::nn::common::MMCVGridSampleParam param)
{
    int block_size    = 256;
    int out_n         = output_shape->GetDim(0);
    int out_c         = output_shape->GetDim(1);
    int out_h         = output_shape->GetDim(2);
    int out_w         = output_shape->GetDim(3);
    int in_h          = input0_shape->GetDim(2);
    int in_w          = input0_shape->GetDim(3);
    int64_t num_elems = output_shape->GetElementsIncludingPadding();
    int grid_size     = DivUp(num_elems, block_size);
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_gridsample_fp32<<<grid_size, block_size, 0, stream>>>(
                out_n, out_c, out_h, out_w, in_h, in_w, num_elems, (float*)input0, (float*)input1, (float*)output, param.align_corners, param.padding_mode, param.interpolation_mode);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_gridsample_fp16<<<grid_size, block_size, 0, stream>>>(
                out_n, out_c, out_h, out_w, in_h, in_w, num_elems, (half*)input0, (half*)input1, (half*)output, param.align_corners, param.padding_mode, param.interpolation_mode);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
