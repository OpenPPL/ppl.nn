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

#include "cudakernel/nn/mmcv_roialign.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/common/tensor_shape.h"
#include <cuda_fp16.h>

template <typename T>
__device__ T bilinear_interpolate(
    const T* input,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/)
{
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width)
        return 0;

    if (y <= 0)
        y = 0;
    if (x <= 0)
        x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y              = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x              = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = input[y_low * width + x_low];
    T v2 = input[y_low * width + x_high];
    T v3 = input[y_high * width + x_low];
    T v4 = input[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename T>
__global__ void mmcv_roi_align_forward_cuda_kernel(
    const int nthreads,
    const T* input,
    const T* rois,
    T* output,
    // T* argmax_y, T* argmax_x,
    const int pooled_height,
    const int pooled_width,
    const T spatial_scale,
    const int sampling_ratio,
    const int pool_mode, // 0 - avg pool, 1 - max pool
    const bool aligned,
    const int channels,
    const int height,
    const int width)
{
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < nthreads;
         index += blockDim.x * gridDim.x) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c  = (index / pooled_width / pooled_height) % channels;
        int n  = index / pooled_width / pooled_height / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind    = offset_rois[0];

        // Do not using rounding; this implementation detail is critical
        T offset      = aligned ? (T)0.5 : (T)0.0;
        T roi_start_w = offset_rois[1] * spatial_scale - offset;
        T roi_start_h = offset_rois[2] * spatial_scale - offset;
        T roi_end_w   = offset_rois[3] * spatial_scale - offset;
        T roi_end_h   = offset_rois[4] * spatial_scale - offset;

        T roi_width  = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
        if (!aligned) { // for backward-compatibility only
            roi_width  = max(roi_width, (T)1.);
            roi_height = max(roi_height, (T)1.);
        }

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        const T* offset_input =
            input + (roi_batch_ind * channels + c) * height * width;

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h =
            (sampling_ratio > 0)
                ? sampling_ratio
                : static_cast<int>(ceil(roi_height / pooled_height));
        int roi_bin_grid_w = (sampling_ratio > 0)
                                 ? sampling_ratio
                                 : static_cast<int>(ceil(roi_width / pooled_width));

        if (pool_mode == 1) {
            // We do max pooling inside a bin
            T maxval = -FLT_MAX;
            // T maxidx_y = -1.f, maxidx_x = -1.f;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                const T y = roi_start_h + ph * bin_size_h +
                            static_cast<T>(iy + .5f) * bin_size_h /
                                static_cast<T>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const T x = roi_start_w + pw * bin_size_w +
                                static_cast<T>(ix + .5f) * bin_size_w /
                                    static_cast<T>(roi_bin_grid_w);
                    T val =
                        bilinear_interpolate(offset_input, height, width, y, x, index);
                    if (val > maxval) {
                        maxval = val;
                        // maxidx_y = y;
                        // maxidx_x = x;
                    }
                }
            }
            output[index] = maxval;
            // argmax_y[index] = maxidx_y;
            // argmax_x[index] = maxidx_x;
        } else if (pool_mode == 0) {
            // We do average pooling inside a bin
            const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
            T output_val  = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                const T y = roi_start_h + ph * bin_size_h +
                            static_cast<T>(iy + .5f) * bin_size_h /
                                static_cast<T>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const T x = roi_start_w + pw * bin_size_w +
                                static_cast<T>(ix + .5f) * bin_size_w /
                                    static_cast<T>(roi_bin_grid_w);
                    T val =
                        bilinear_interpolate(offset_input, height, width, y, x, index);
                    output_val += val;
                }
            }
            output[index] = output_val / count;
        }
    }
}

// onnx version differ with onnx/pytorch:
// 1. batch_indices as an individual input in onnx, but as one dimension in onnx
// 2. roi_cols as an argument in onnx, but as an constant(5) in pytorch
// 3. in max_mode, onnx acquires max_val in interploting process
ppl::common::RetCode PPLCUDAMMCVROIAlignForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* rois_shape,
    const void* rois,
    ppl::nn::TensorShape* output_shape,
    void* output,
    ppl::nn::common::MMCVROIAlignParam param)
{
    int block_size    = 256;
    int channels      = input_shape->GetDim(1);
    int height        = input_shape->GetDim(2);
    int width         = input_shape->GetDim(3);
    int64_t num_elems = output_shape->GetElementsIncludingPadding();
    int grid_size     = DivUp(num_elems, block_size);
    bool is_mode_max  = param.pool_mode != "avg";
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        mmcv_roi_align_forward_cuda_kernel<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (const float*)rois, (float*)output, param.aligned_height, param.aligned_width, param.spatial_scale, param.sampling_ratio, is_mode_max, param.aligned, channels, height, width);
    }
    return ppl::common::RC_SUCCESS;
}
