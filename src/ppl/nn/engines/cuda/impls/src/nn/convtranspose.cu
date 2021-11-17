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

#include "cudakernel/nn/convtranspose.h"
#include "cudakernel/math/math.h"
#include "cudakernel/memory/transpose.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/params/onnx/gemm_param.h"
#include <cuda_fp16.h>

#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

template <typename T>
__global__ void ppl_col2im_gpu_kernel(
    const int n,
    const T* data_col,
    const int height,
    const int width,
    const int channels,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int hole_h,
    const int hole_w,
    const int height_col,
    const int width_col,
    const float beta,
    T* data_im)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        T val               = 0;
        const int w_im      = index % width + pad_w;
        const int h_im      = (index / width) % height + pad_h;
        const int c_im      = index / (width * height);
        int kernel_extern_w = (kernel_w - 1) * hole_w + 1;
        int kernel_extern_h = (kernel_h - 1) * hole_h + 1;
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_extern_w) ? 0 : (w_im - kernel_extern_w) / stride_w + 1;
        const int w_col_end =
            min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extern_h) ? 0 : (h_im - kernel_extern_h) / stride_h + 1;
        const int h_col_end =
            min(h_im / stride_h + 1, height_col);
        // equivalent implementation
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % hole_h == 0 && w_k % hole_w == 0) {
                    h_k /= hole_h;
                    w_k /= hole_w;
                    int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) * width_col + w_col;
                    val                = Math<T, T, T>::add(val, data_col[data_col_index]);
                }
            }
        }
        data_im[index] = Math<T, T, T>::add(val, (beta == 0 ? (T)0 : data_im[index]));
    }
}

template <typename T>
void ppl_col2im_gpu(
    cudaStream_t stream,
    const T* data_col,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int hole_h,
    int hole_w,
    int height_col,
    int width_col,
    const float beta,
    T* data_im)
{
    int num_kernels = channels * height * width;
    ppl_col2im_gpu_kernel<T><<<(num_kernels + 1024 - 1) / 1024, 1024, 0, stream>>>(
        num_kernels, data_col, height, width, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w, height_col, width_col, beta, data_im);
}

template <typename T>
__global__ void ppl_cukernel_matrix_padding(
    const T* input,
    int inHeight,
    int inWidth,
    T* output,
    int out_height,
    int out_width)
{
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (w_idx >= inWidth || h_idx >= inHeight)
        return;
    uint64_t in_index  = h_idx * inWidth + w_idx;
    uint64_t out_index = h_idx * out_width + w_idx;
    output[out_index]  = input[in_index];
}

template <typename T>
void cuda_matrix_padding(
    cudaStream_t stream,
    const T* input,
    int inHeight,
    int inWidth,
    T* output,
    int out_height,
    int out_width)
{
    cudaMemset(output, 0, out_height * out_width * sizeof(T));
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((inWidth + 15) / 16, (inHeight + 15) / 16, 1);
    ppl_cukernel_matrix_padding<T><<<gridSize, blockSize, 0, stream>>>(
        input, inHeight, inWidth, output, out_height, out_width);
}

template <typename T>
void __global__ addVectorToMatrixColumnKernel(
    int numRows,
    int numCols,
    int stride,
    float alpha,
    const T* bias,
    float beta,
    T* out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numRows * numCols)
        return;
    int row_idx        = index / numCols;
    int col_idx        = index % numCols;
    uint64_t out_index = row_idx * stride + col_idx;
    uint64_t in_index  = index;

    out[out_index] = Math<T, T, T>::add(
        Math<T, T, T>::mul((T)alpha, bias[row_idx]),
        ((beta == 0) ? (T)0 : Math<T, T, T>::mul((T)beta, out[in_index])));
}

template <typename T>
void addVectorToMatrixColumn(
    cudaStream_t stream,
    int M,
    int N,
    int stride,
    float alpha,
    const T* biasData,
    float beta,
    T* outData)
{
    const uint64_t count = M * N;
    const int blockSize  = 128;
    const int gridSize   = (count + blockSize - 1) / blockSize;
    addVectorToMatrixColumnKernel<T><<<gridSize, blockSize, 0, stream>>>(M, N, stride, alpha, biasData, beta, outData);
}

uint64_t pplConvTransposeGetFilterBufSizeCudaFp32(
    const int num_filters,
    const int num_channels,
    const int filter_height,
    const int filter_width)
{
    int M = num_filters;
    M *= filter_height;
    M *= filter_width;
    size_t K = num_channels;

    size_t padM = Align(M, 1);
    size_t padK = Align(K, 8);

    size_t dst = Align(padM * padK * sizeof(__half), 128);
    return dst * 2; // transpose buf
}

template <typename T>
void pplConvTransposeConvertFilter(
    cudaStream_t stream,
    const T* filter,
    int num_filters,
    int num_channels,
    int filter_height,
    int filter_width,
    T* cvt_filter)
{
    int M = num_filters;
    M *= filter_height;
    M *= filter_width;
    size_t K = num_channels;

    size_t padM = Align(M, 1);
    size_t padK = Align(K, 8);
    // no need to transpose, just
    cuda_matrix_padding<T>(stream, filter, K, M, cvt_filter, padK, padM);
}

uint64_t pplConvTransposeGetTempBufSizeCudaFp32(
    const int group,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_c,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int hole_h,
    const int hole_w)
{
    size_t M = out_c * kernel_h * kernel_w;
    size_t N = in_w * in_h;
    size_t K = in_c;

    size_t padN = Align(N, 8);
    size_t padK = Align(K, 8);

    // for trans buf
    return 2 * Align(padN * padK * sizeof(__half), 128) +
           Align(M * N * sizeof(__half), 128) + Align(M * padN * sizeof(__half), 128);
}

uint64_t PPLConvTransposeGetBufSizeCuda(
    ppl::nn::TensorShape* input_shape,
    ppl::nn::TensorShape* output_shape,
    const ppl::nn::common::ConvTransposeParam* param)
{
    int batch     = input_shape->GetDim(0);
    int in_c      = input_shape->GetDim(1);
    int in_h      = input_shape->GetDim(2);
    int in_w      = input_shape->GetDim(3);
    int out_c     = output_shape->GetDim(1);
    int out_h     = output_shape->GetDim(2);
    int out_w     = output_shape->GetDim(3);
    int group     = param->group;
    int kernel_h  = param->kernel_shape[0];
    int kernel_w  = param->kernel_shape[1];
    int pad_h     = param->pads[0];
    int pad_w     = param->pads[1];
    int stride_h  = param->strides[0];
    int stride_w  = param->strides[1];
    int hole_h    = param->dilations[0];
    int hole_w    = param->dilations[1];
    uint64_t size = 0;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        size += pplConvTransposeGetFilterBufSizeCudaFp32(out_c, in_c, kernel_h, kernel_w) +
                pplConvTransposeGetTempBufSizeCudaFp32(group, in_c, in_h, in_w, out_c, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w);
    } else {
        return 0;
    }

    // NT gemm
    int transA = 0;
    int M      = out_c * kernel_h * kernel_w;
    int K      = in_c;
    int padM   = Align(M, 1);
    int padK   = Align(K, 8);
    ppl::nn::TensorShape a_shape;
    a_shape.Reshape({padM, padK});
    a_shape.SetDataType(input_shape->GetDataType());
    size += PPLGemmCUDAGetBufSize(&a_shape, transA);

    return size;
}

template <typename T>
__global__ void remove_padding(
    T* pad_data,
    T* data,
    const int M,
    const int padN,
    const int N)
{
    int64_t off    = blockIdx.x * 256 + threadIdx.x;
    int m_id       = off / N;
    int n_id       = off % N;
    int64_t in_off = (int64_t)m_id * padN + n_id;

    if (off < M * N)
        data[off] = pad_data[in_off];
}
template <typename T>
T* RemovePadding(
    const cudaStream_t& stream,
    T* pad_data,
    T* data,
    const int M,
    const int padN,
    const int N)
{
    if (padN == N)
        return pad_data;
    int block_size = 256;
    int grid       = (M * N + 255) / 256;
    remove_padding<T><<<grid, block_size, 0, stream>>>(pad_data, data, M, padN, N);
    return data;
}

ppl::common::RetCode PPLCUDAConvTransposeCvt(
    cudaStream_t stream,
    const void* in_filter,
    void* temp_buffer,
    void* out_filter,
    const ppl::nn::TensorShape* filter_shape,
    const ppl::nn::common::ConvTransposeParam* param)
{
    int in_c     = filter_shape->GetDim(1);
    int out_c    = filter_shape->GetDim(0);
    int kernel_h = param->kernel_shape[0];
    int kernel_w = param->kernel_shape[1];
    int pad_h    = param->pads[0];
    int pad_w    = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int hole_h   = param->dilations[0];
    int hole_w   = param->dilations[1];
    int num_channels     = in_c;
    int num_filters      = out_c;
    int M                = out_c * kernel_h * kernel_w;
    int K                = in_c;
    int padM             = Align(M, 1);
    int padK             = Align(K, 8);

    // cvt filter
    __half* cvt_filter   = (__half*)temp_buffer;
    pplConvTransposeConvertFilter<__half>(stream, (__half*)in_filter, num_filters, num_channels, kernel_h, kernel_w, cvt_filter);

    ppl::nn::common::TransposeParam trans_param;
    trans_param.perm.push_back(1);
    trans_param.perm.push_back(0);

    ppl::nn::TensorShape a_shape, out_a_shape;
    a_shape.SetDataType(filter_shape->GetDataType());
    out_a_shape.SetDataType(filter_shape->GetDataType());
    a_shape.Reshape({padK, M});
    out_a_shape.Reshape({padM, padK});

    ppl::common::RetCode status = PPLCUDATransposeForwardImp(stream,
                                                             trans_param,
                                                             &a_shape,
                                                             cvt_filter,
                                                             &out_a_shape,
                                                             out_filter);
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAConvTransposeForward(
    cudaStream_t stream,
    ppl::nn::cuda::CUDAModule* module,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const void* trans_filter,
    const void* bias,
    const ppl::nn::common::ConvTransposeParam* param,
    algo_param_t algo_param,
    void* temp_buffer,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    int batch    = input_shape->GetDim(0);
    int in_c     = input_shape->GetDim(1);
    int in_h     = input_shape->GetDim(2);
    int in_w     = input_shape->GetDim(3);
    int out_c    = output_shape->GetDim(1);
    int out_h    = output_shape->GetDim(2);
    int out_w    = output_shape->GetDim(3);
    int kernel_h = param->kernel_shape[0];
    int kernel_w = param->kernel_shape[1];
    int pad_h    = param->pads[0];
    int pad_w    = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int hole_h   = param->dilations[0];
    int hole_w   = param->dilations[1];

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        int num_channels     = in_c;
        int num_filters      = out_c;
        int height           = in_h;
        int width            = in_w;
        int out_height       = out_h;
        int out_width        = out_w;
        int M                = out_c * kernel_h * kernel_w;
        int N                = in_w * in_h;
        int K                = in_c;
        int padM             = Align(M, 1);
        int padN             = Align(N, 8);
        int padK             = Align(K, 8);
        __half* pad_in_data  = (__half*)temp_buffer;
        __half* pad_out_data = pad_in_data + Align(padN * padK, 128 / sizeof(__half));
        __half* out_data     = pad_out_data + Align(M * padN, 128 / sizeof(__half));

        ppl::nn::common::TransposeParam trans_param;
        trans_param.perm.push_back(1);
        trans_param.perm.push_back(0);
        __half* trans_in_data = out_data + Align(M * N, 128 / sizeof(__half));

        ppl::nn::TensorShape a_shape, b_shape, c_shape, out_a_shape, out_b_shape;
        a_shape.SetDataType(input_shape->GetDataType());
        b_shape.SetDataType(input_shape->GetDataType());
        c_shape.SetDataType(output_shape->GetDataType());
        out_a_shape.SetDataType(input_shape->GetDataType());
        out_b_shape.SetDataType(input_shape->GetDataType());
        a_shape.Reshape({padK, M});
        b_shape.Reshape({padK, padN});
        c_shape.Reshape({M, padN});
        out_a_shape.Reshape({padM, padK});
        out_b_shape.Reshape({padN, padK});

        ppl::nn::common::GemmParam gemm_param;
        fuse_param_t fuse_param;
        gemm_param.bias_term = 0;
        gemm_param.transA    = 0;
        gemm_param.transB    = 1;
        gemm_param.alpha     = 1.f;
        gemm_param.beta      = 1.f;
        gemm_param.N         = padN;
        for (int n = 0; n < batch; ++n) {
            int offset_in  = n * num_channels * height * width;
            int offset_out = n * num_filters * out_height * out_width;
            cuda_matrix_padding<__half>(stream, ((__half*)input) + offset_in, K, N, pad_in_data, padK, padN);

            PPLCUDATransposeForwardImp(stream,
                                       trans_param,
                                       &b_shape,
                                       pad_in_data,
                                       &out_b_shape,
                                       trans_in_data);

            // NT
            ppl::nn::TensorShape a_shape, b_shape, c_shape;
            // input transpose KxN -> NxK    weight transpose KxM -> MxK
            PPLCUDAGemmForwardImp(stream, module, &out_a_shape, trans_filter, &out_b_shape, trans_in_data, NULL, &c_shape, pad_out_data, gemm_param, NULL, fuse_param, algo_param);

            __half* tmp = RemovePadding<__half>(stream, pad_out_data, out_data, M, padN, N);
            ppl_col2im_gpu<__half>(stream, (const __half*)tmp, num_filters, out_height, out_width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w, height, width, 0.f, ((__half*)output) + offset_out);

            if (NULL != bias) {
                addVectorToMatrixColumn<__half>(stream, num_filters, out_height * out_width, out_height * out_width, 1.f, (__half*)bias, 1.f, ((__half*)output) + offset_out);
            }
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
