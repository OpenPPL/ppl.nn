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

#include "cudakernel/nn/resize.h"
#include "ppl/common/types.h"

struct half8_ {
    half x0;
    half y0;
    half z0;
    half w0;
    half x1;
    half y1;
    half z1;
    half w1;
};

static inline __device__ float cudaComputeSourceIndexCubic(
    float scale,
    int dstIndex)
{
    float srcIdx = scale * (dstIndex + 0.5) - 0.5;
    return srcIdx;
}

static inline __device__ float cudaComputeSourceIndexNearest(
    float scale,
    int dstIndex,
    int transform_mode)
{
    float srcIdx = 0.f;
    if (transform_mode == 3) {
        srcIdx = scale * dstIndex;
    } else {
        srcIdx = scale * (dstIndex + 0.5) - 0.5;
    }
    return (srcIdx < 0) ? 0.f : srcIdx;
}

static inline __device__ float cudaComputeSourceIndexBilinear(
    float scale,
    int dstIndex)
{
    float srcIdx = scale * (dstIndex + 0.5) - 0.5;
    return (srcIdx < 0) ? 0.f : srcIdx;
}

static __device__ __forceinline__ float cubic_convolution1(
    float x,
    float A)
{
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

static __device__ __forceinline__ float cubic_convolution2(
    float x,
    float A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

static __device__ __forceinline__ void get_cubic_resize_coefficients(
    float coeffs[4],
    float t,
    float cubic_coeff)
{
    float A = cubic_coeff;

    float x1  = t;
    coeffs[0] = cubic_convolution2(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1(x1, A);

    // opposite coefficients
    float x2  = 1.0 - t;
    coeffs[2] = cubic_convolution1(x2, A);
    coeffs[3] = cubic_convolution2(x2 + 1.0, A);
}

template <typename T>
static __device__ inline T cubic_interplote(float frac0, T data0, float frac1, T data1, float frac2, T data2, float frac3, T data3);

template <typename T>
static __device__ inline T cubic_interplote(float frac0, T data0, float frac1, T data1, float frac2, T data2, float frac3, T data3)
{
    T res;
    res = frac0 * data0 + frac1 * data1 +
          frac2 * data2 + frac3 * data3;
    return res;
}

template <>
__device__ inline half cubic_interplote<half>(float frac0, half data0, float frac1, half data1, float frac2, half data2, float frac3, half data3)
{
    half res;
    res = frac0 * __half2float(data0) + frac1 * __half2float(data1) +
          frac2 * __half2float(data2) + frac3 * __half2float(data3);
    return res;
}

// template <>
// __device__ inline half8_ cubic_interplote<half8_>(float frac0, half8_ data0, float frac1, half8_ data1, float frac2, half8_ data2, float frac3, half8_ data3)
// {
//     half8_ res;
//     res.x0 = frac0 * __half2float(data0.x0) + frac1 * __half2float(data1.x0) +
//              frac2 * __half2float(data2.x0) + frac3 * __half2float(data3.x0);
//     res.y0 = frac0 * __half2float(data0.y0) + frac1 * __half2float(data1.y0) +
//              frac2 * __half2float(data2.y0) + frac3 * __half2float(data3.y0);
//     res.z0 = frac0 * __half2float(data0.z0) + frac1 * __half2float(data1.z0) +
//              frac2 * __half2float(data2.z0) + frac3 * __half2float(data3.z0);
//     res.w0 = frac0 * __half2float(data0.w0) + frac1 * __half2float(data1.w0) +
//              frac2 * __half2float(data2.w0) + frac3 * __half2float(data3.w0);
//     res.x1 = frac0 * __half2float(data0.x1) + frac1 * __half2float(data1.x1) +
//              frac2 * __half2float(data2.x1) + frac3 * __half2float(data3.x1);
//     res.y1 = frac0 * __half2float(data0.y1) + frac1 * __half2float(data1.y1) +
//              frac2 * __half2float(data2.y1) + frac3 * __half2float(data3.y1);
//     res.z1 = frac0 * __half2float(data0.z1) + frac1 * __half2float(data1.z1) +
//              frac2 * __half2float(data2.z1) + frac3 * __half2float(data3.z1);
//     res.w1 = frac0 * __half2float(data0.w1) + frac1 * __half2float(data1.w1) +
//              frac2 * __half2float(data2.w1) + frac3 * __half2float(data3.w1);
//     return res;
// }

template <typename T>
static __device__ __forceinline__ T cubic_interp1d(
    T x0,
    T x1,
    T x2,
    T x3,
    float t,
    float cubic_coeff)
{
    float coeffs[4];
    get_cubic_resize_coefficients(coeffs, t, cubic_coeff);

    return cubic_interplote<T>(coeffs[0], x0, coeffs[1], x1, coeffs[2], x2, coeffs[3], x3);
}

template <typename T>
__device__ __forceinline__ static T resize_get_value_bounded(
    const T* data,
    int height,
    int width,
    int access_c,
    int y,
    int x)
{
    int access_y = max(min(y, height - 1), 0);
    int access_x = max(min(x, width - 1), 0);
    return data[access_c * height * width + access_y * width + access_x];
}

template <typename T>
__device__ inline T bilinear_interplote(float frac_w0, float frac_w1, float frac_h0, float frac_h1, T data0, T data1, T data2, T data3)
{
    T res;
    res = frac_h0 * (frac_w0 * data0 + frac_w1 * data1) +
          frac_h1 * (frac_w0 * data2 + frac_w1 * data3);
    return res;
}

template <>
__device__ inline half bilinear_interplote<half>(float frac_w0, float frac_w1, float frac_h0, float frac_h1, half data0, half data1, half data2, half data3)
{
    half res;
    res = frac_h0 * (frac_w0 * __half2float(data0) + frac_w1 * __half2float(data1)) +
          frac_h1 * (frac_w0 * __half2float(data2) + frac_w1 * __half2float(data3));
    return res;
}

template <>
__device__ inline half8_ bilinear_interplote<half8_>(float frac_w0, float frac_w1, float frac_h0, float frac_h1, half8_ data0, half8_ data1, half8_ data2, half8_ data3)
{
    half8_ res;
    res.x0 = frac_h0 * (frac_w0 * __half2float(data0.x0) + frac_w1 * __half2float(data1.x0)) +
             frac_h1 * (frac_w0 * __half2float(data2.x0) + frac_w1 * __half2float(data3.x0));
    res.y0 = frac_h0 * (frac_w0 * __half2float(data0.y0) + frac_w1 * __half2float(data1.y0)) +
             frac_h1 * (frac_w0 * __half2float(data2.y0) + frac_w1 * __half2float(data3.y0));
    res.z0 = frac_h0 * (frac_w0 * __half2float(data0.z0) + frac_w1 * __half2float(data1.z0)) +
             frac_h1 * (frac_w0 * __half2float(data2.z0) + frac_w1 * __half2float(data3.z0));
    res.w0 = frac_h0 * (frac_w0 * __half2float(data0.w0) + frac_w1 * __half2float(data1.w0)) +
             frac_h1 * (frac_w0 * __half2float(data2.w0) + frac_w1 * __half2float(data3.w0));
    res.x1 = frac_h0 * (frac_w0 * __half2float(data0.x1) + frac_w1 * __half2float(data1.x1)) +
             frac_h1 * (frac_w0 * __half2float(data2.x1) + frac_w1 * __half2float(data3.x1));
    res.y1 = frac_h0 * (frac_w0 * __half2float(data0.y1) + frac_w1 * __half2float(data1.y1)) +
             frac_h1 * (frac_w0 * __half2float(data2.y1) + frac_w1 * __half2float(data3.y1));
    res.z1 = frac_h0 * (frac_w0 * __half2float(data0.z1) + frac_w1 * __half2float(data1.z1)) +
             frac_h1 * (frac_w0 * __half2float(data2.z1) + frac_w1 * __half2float(data3.z1));
    res.w1 = frac_h0 * (frac_w0 * __half2float(data0.w1) + frac_w1 * __half2float(data1.w1)) +
             frac_h1 * (frac_w0 * __half2float(data2.w1) + frac_w1 * __half2float(data3.w1));
    return res;
}

template <typename T>
__global__ void ppl_cukernel_resize_bilinear(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1
        // special case: just copy
        if (in_height == out_height && in_width == out_width) {
            const int h1  = h2;
            const int w1  = w2;
            const T* pos1 = &input[h1 * in_width + w1];
            T* pos2       = &output[h2 * out_width + w2];
            for (int c = 0; c < channels; ++c) {
                pos2[0] = pos1[0];
                pos1 += in_width * in_height;
                pos2 += out_width * out_height;
            }
            return;
        }

        // const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndexBilinear(h_scale, h2);

        const int h1         = h1r;
        const int h1p        = (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;

        // const float w1r = w_scale * w2;
        const float w1r      = cudaComputeSourceIndexBilinear(w_scale, w2);
        const int w1         = w1r;
        const int w1p        = (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;

        const T* pos1 = &input[h1 * in_width + w1];
        T* pos2       = &output[h2 * out_width + w2];
        for (int c = 0; c < channels; ++c) {
            // pos2[0] = h0lambda * (w0lambda * pos1[0] +
            // w1lambda * pos1[w1p]) +
            // h1lambda * (w0lambda * pos1[h1p * in_width] +
            // w1lambda * pos1[h1p * in_width + w1p]);
            pos2[0] = bilinear_interplote<T>(w0lambda, w1lambda, h0lambda, h1lambda, pos1[0], pos1[w1p], pos1[h1p * in_width], pos1[h1p * in_width + w1p]);
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1
        // special case: just copy
        if (in_height == out_height && in_width == out_width) {
            const int h1  = h2;
            const int w1  = w2;
            const T* pos1 = &input[h1 * in_width + w1];
            T* pos2       = &output[h2 * out_width + w2];
            for (int c = 0; c < channels; ++c) {
                pos2[0] = pos1[0];
                pos1 += in_width * in_height;
                pos2 += out_width * out_height;
            }
            return;
        }

        // const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndexNearest(h_scale, h2, transform_mode);
        const int h1    = h1r;

        // const float w1r = w_scale * w2;
        const float w1r = cudaComputeSourceIndexNearest(w_scale, w2, transform_mode);
        const int w1    = w1r;

        const T* pos1 = &input[h1 * in_width + w1];
        T* pos2       = &output[h2 * out_width + w2];
        for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_cubic(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float cubic_coeff)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1
        // special case: just copy
        if (in_height == out_height && in_width == out_width) {
            const int h1  = h2;
            const int w1  = w2;
            const T* pos1 = &input[h1 * in_width + w1];
            T* pos2       = &output[h2 * out_width + w2];
            for (int c = 0; c < channels; ++c) {
                pos2[0] = pos1[0];
                pos1 += in_width * in_height;
                pos2 += out_width * out_height;
            }
            return;
        }

        const float h1r      = cudaComputeSourceIndexCubic(h_scale, h2);
        const int h1         = floorf(h1r);
        const float h1lambda = h1r - h1;

        const float w1r      = cudaComputeSourceIndexCubic(w_scale, w2);
        const int w1         = floorf(w1r);
        const float w1lambda = w1r - w1;

        T* pos2 = &output[h2 * out_width + w2];
        for (int c = 0; c < channels; ++c) {
            T coefficients[4];

            for (int k = 0; k < 4; k++) {
                coefficients[k] = cubic_interp1d<T>(
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 - 1),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 0),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 1),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 2),
                    w1lambda,
                    cubic_coeff);
            }
            pos2[0] = cubic_interp1d<T>(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                h1lambda,
                cubic_coeff);

            pos2 += out_width * out_height;
        }
    }
}

static inline float hostComputeAreaScale(int input_size, int output_size, int mode)
{
    if (output_size > 1 || mode == 0 || mode == 3) {
        return float(input_size) / output_size;
    } else {
        return 0.f;
    }
}
// coordinate_transformation_mode definition
// {"half_pixel", 0}, {"pytorch_half_pixel", 1}, {"align_corners", 2},
// {"asymmetric", 3}, {"tf_half_pixel_for_nn", 4}, {"tf_crop_and_resize", 5}
// interpolation mode
// {"nearest", 0}, {"linear", 1}, {"cubic", 2}
template <typename T>
void ppl_resize_forward(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const T* input,
    const ppl::nn::TensorShape* output_shape,
    T* output,
    bool scale_pre_set,
    float h_scale_pre,
    float w_scale_pre,
    int transform_mode,
    int inter_mode,
    float cubic_coeff)
{
    int dim_count  = output_shape->GetDimCount();
    int out_height = 1, out_width = 1;
    int in_height = 1, in_width = 1;
    for (int it = 2; it < dim_count - 1; ++it) {
        out_height *= output_shape->GetDim(it);
        in_height *= input_shape->GetDim(it);
    }
    out_width    = output_shape->GetDim(dim_count - 1);
    in_width     = input_shape->GetDim(dim_count - 1);
    int channels = output_shape->GetDim(0) * output_shape->GetDim(1);

    float h_scale = 0.f, w_scale = 0.f;
    if (scale_pre_set) {
        h_scale = h_scale_pre;
        w_scale = w_scale_pre;
    } else {
        h_scale = hostComputeAreaScale(in_height, out_height, transform_mode);
        w_scale = hostComputeAreaScale(in_width, out_width, transform_mode);
    }
    int num_threads = out_height * out_width;
    int block_size  = 256;
    int grid        = (num_threads + block_size - 1) / block_size;
    if (inter_mode == 0) {
        ppl_cukernel_resize_nearest<T><<<grid, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, input, in_height, in_width, output, out_height, out_width, transform_mode);
    } else if (inter_mode == 1) {
        ppl_cukernel_resize_bilinear<T><<<grid, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, input, in_height, in_width, output, out_height, out_width);
    } else if (inter_mode == 2) {
        ppl_cukernel_resize_cubic<T><<<grid, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, input, in_height, in_width, output, out_height, out_width, cubic_coeff);
    }
}

ppl::common::RetCode PPLCUDAResizeForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    bool scale_pre_set,
    float h_scale,
    float w_scale,
    int transform_mode,
    int inter_mode,
    float cubic_coeff)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            ppl_resize_forward(stream, input_shape, (const half*)input, output_shape, (half*)output, scale_pre_set, h_scale, w_scale, transform_mode, inter_mode, cubic_coeff);
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl_resize_forward(stream, input_shape, (const float*)input, output_shape, (float*)output, scale_pre_set, h_scale, w_scale, transform_mode, inter_mode, cubic_coeff);

    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
