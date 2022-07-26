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

#include "cudakernel/gemm/gemm.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "cudakernel/common/cuda_check.h"

#include <cuda_fp16.h>
#include <float.h>
#include <algorithm>

#include "kernel_type.h"
#include "conv_common.h"
#include "cudakernel/nn/conv/gene_kernel.h"

#define TIMES 4

static std::vector<kernel_info_t> g_int8_kvec;
static bool is_g_int8_kvec_set = false;

#define FAKE_INT8_CONV_PARAM              \
    int in_hw               = 1;     \
    int out_hw              = 1;     \
    int flt_hw              = 1;     \
    int splitk              = 1;     \
    int in_height           = 1;     \
    int in_width            = 1;     \
    int batch               = M;     \
    int num_grp             = 1;     \
    int num_chl_per_grp     = 0;     \
    int num_chl_per_grp_pad = K_pad; \
    int flt_height          = 1;     \
    int flt_width           = 1;     \
    int num_flt_per_grp     = N;     \
    int num_flt_per_grp_pad = N_pad; \
    int out_height          = 1;     \
    int out_width           = 1;     \
    int stride_height       = 1;     \
    int stride_width        = 1;     \
    int pad_height          = 0;     \
    int pad_width           = 0;     \
    int hole_height         = 1;     \
    int hole_width          = 1;

#define INT8_GEMM_FUNC_PARAM                                                   \
        input0_tmp,                                                        \
        (int4 *)weight,                                                    \
        final_out,                                                         \
        kLoopNum,                                                          \
        in_lut,                        0,                                  \
        flt_lut,                       0,                                  \
        in_hw,                         out_hw,                             \
        flt_hw,                        splitk,                             \
        in_height,                     in_width,                           \
        batch,                         num_grp,                            \
        num_chl_per_grp,               num_chl_per_grp_pad,                \
        flt_height,                    flt_width,                          \
        num_flt_per_grp,               num_flt_per_grp_pad,                \
        out_height,                    out_width,                          \
        stride_height,                 stride_width,                       \
        pad_height,                    pad_width,                          \
        hole_height,                   hole_width,                         \
        has_bias,                      (int4 *)bias,                       \
        alpha*quant_param.in_scale,    quant_param.d_flt_scale,            \
        quant_param.out_scale,         quant_param.pre_scale,              \
        fuse_param.has_activation,     fuse_param.clip_min,                \
        fuse_param.has_clip,           fuse_param.clip_max,                \
        fuse_param.has_prelu,          (const void *)fuse_param.prelu,     \
        fuse_param.has_elt,            (const int4 *)fuse_param.pre_data,  \
        fuse_param.has_elt_activation, fuse_param.elt_clip_min,            \
        fuse_param.has_elt_clip,       fuse_param.elt_clip_max,            \
        fuse_param.has_elt_prelu,      (const void *)fuse_param.elt_prelu, \
        (__half)fuse_param.leaky,      (__half)fuse_param.elt_leaky,       \
        fuse_param.has_concat,         concat_offset_v8,                   \
        concat_stride_v8

void init_f1_int8_kvec(std::vector<kernel_info_t> &g_int8_kvec, int device_id, ppl::common::datatype_t type)
{
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

#ifndef PPLNN_ENABLE_CUDA_JIT
    if(type == ppl::common::DATATYPE_INT8) {
        if (device_prop.major == 7 && device_prop.minor == 5) {
            Initialize2spkSM75Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
        } else if (device_prop.major > 8 || (device_prop.major == 8 && device_prop.minor >= 0)) {
            Initialize2spkSM75Int8Imma8816ConvF1KernelContainer(g_int8_kvec);

            Initialize2spkSM80Int8Imma8816ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16816ConvF1KernelContainer(g_int8_kvec);
            Initialize2spkSM80Int8Imma16832ConvF1KernelContainer(g_int8_kvec);
        }
    }
    is_g_int8_kvec_set = true;
#endif
}

// block size: (32,32,1)
__global__ void int8_matrix_transpose(
    int *output,
    int *input,
    const int in_row,
    const int in_col)
{
    unsigned int in_x  = blockIdx.x * 32 + threadIdx.x;
    unsigned int in_y  = blockIdx.y * 32 + threadIdx.y;
    unsigned int out_x = blockIdx.y * 32 + threadIdx.x;
    unsigned int out_y = blockIdx.x * 32 + threadIdx.y;
    bool in_range      = (in_x <= in_col) && (in_y <= in_row);
    bool out_range     = (out_x <= in_row) && (out_y <= in_col);
    __shared__ int smem[32][33];

    int value                        = in_range ? input[in_y * in_col + in_x] : (int)0;
    smem[threadIdx.x][threadIdx.y] = value;

    __syncthreads();
    value          = smem[threadIdx.y][threadIdx.x];
    if (out_range)
        output[out_y * in_row + out_x] = value;
}

ppl::common::RetCode PPLCUDAGemmModifyWeightsInt8(
    const cudaStream_t &stream,
    ppl::nn::TensorShape *weight_shape,
    void *weight,
    void *tmp_weight, // if need transpose
    const ppl::nn::onnx::GemmParam *param)
{
    int transB   = param->transB;
    auto type    = weight_shape->GetDataType();
    int pad_size = GetPadSize(type);

    const int dim0 = weight_shape->GetDim(0); // assume padded
    const int dim1 = weight_shape->GetDim(1);

    if (!transB) {
#define TRANSWEIGHT                                                                                     \
    int8_matrix_transpose<<<grid, block, 0, stream>>>((int *)tmp_weight, (int *)weight, dim0, dim1); \
    cudaMemcpyAsync((int8_t *)weight, (int8_t *)tmp_weight, dim0 *dim1 * sizeof(int8_t), cudaMemcpyDeviceToDevice, stream);

        dim3 grid(DivUp(dim1, 32), DivUp(dim0, 32), 1);
        dim3 block(32, 32, 1);
        weight_shape->SetDim(0, dim1);
        weight_shape->SetDim(1, dim0);
        if (type == ppl::common::DATATYPE_INT8) {
            TRANSWEIGHT
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
#undef TRANSWEIGHT
    }
    return ppl::common::RC_SUCCESS;
}

double PPLCUDAGemmJITSelectKernelInt8(
   int device_id,
   cudaStream_t &stream,
   ppl::common::datatype_t type,
   ppl::nn::TensorShape *input_shape,
   void *input,
   ppl::nn::TensorShape *weight_shape,
   void *weight,
   void *bias,
   ppl::nn::TensorShape *output_shape,
   void *output,
   void *temp_buffer,
   conv_param_t &conv_param,
   quant_param_t &quant_param,
   fuse_param_t &fuse_param,
   algo_param_t &algo_param,
   uint64_t workspace)
{
   double elapsed = 0.0f;
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::vector<std::string> knames;
    std::vector<algo_param_t> params;
    std::string sources = "";

    GetInt8ConvKernelNominees(device_id, type, conv_param, knames, params, sources);

    int index = 0;
    std::vector<const char *> compile_params;
    elapsed = AlgoForwardTimeInt8(device_id, stream, knames, sources, index, compile_params, device_id, true, type, (int4 *)input, (int4 *)weight, (int4 *)output, (int4 *)bias, (int4 *)temp_buffer, params, conv_param, quant_param, fuse_param, workspace);

    algo_param = params[index];
#endif
    return elapsed;
}

double PPLCUDAGemmSelectKernelInt8(
    int device_id,
    const cudaStream_t &stream,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *weight_shape,
    const void *weight,
    const void *bias,
    const ppl::nn::TensorShape *output_shape,
    void *output,
    void *temp_buffer,
    const ppl::nn::onnx::GemmParam &param,
    const quant_param_t &quant_param,
    const fuse_param_t &fuse_param,
    algo_param_t &algo_param)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    auto type = weight_shape->GetDataType();
    if (!is_g_int8_kvec_set)
        init_f1_int8_kvec(g_int8_kvec, device_id, type);

    int pad_size = GetPadSize(type);
    int transA   = param.transA;
    int transB   = param.transB;
    float alpha  = param.alpha;

    // FIXME use non-paded N in conv1x1 for input
    int N     = transB ? weight_shape->GetDim(0) : weight_shape->GetDim(1);
    int N_pad = transB ? weight_shape->GetDim(0) : weight_shape->GetDim(1);
    int K_pad = transB ? weight_shape->GetDim(1) : weight_shape->GetDim(0);
    int M     = transA ? input_shape->GetDim(1) : input_shape->GetDim(0);

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;
    int4 *final_out      = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : (int4 *)output;

    // fuse configs
    bool has_bias        = bias; // beta != 0.f;

    float minTime = FLT_MAX;
    int best_kid  = -1;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // transpose
    int4 *input0_tmp = (int4 *)input;
    if (transA == 1) { // input is shape of (K, M), we need K as the 1st inner dim
        int K_pad_v4 = K_pad / 4;
        dim3 grid(DivUp(K_pad_v4, 32), DivUp(M, 32), 1);
        dim3 block(32, 32, 1);
        int8_matrix_transpose<<<grid, block, 0, stream>>>((int*)temp_buffer, (int*)input, K_pad_v4, M);
        input0_tmp = (int4 *)temp_buffer;
    }

    for (unsigned int kid = 0; kid < g_int8_kvec.size(); kid++) {
        int tile_m_per_cta = g_int8_kvec[kid].tile_m_per_cta;
        int tile_n_per_cta = g_int8_kvec[kid].tile_n_per_cta;
        int tile_k_per_cta = g_int8_kvec[kid].tile_k_per_cta;
        int cta_size_in_thd = g_int8_kvec[kid].cta_size_in_thd;
        int smem_size = g_int8_kvec[kid].smem_size;

        if (!g_int8_kvec[kid].CheckSMemSizeFeasible(device_prop))
                continue;

        if (!g_int8_kvec[kid].CheckGpuArchFeasible(device_prop))
                continue;

        g_int8_kvec[kid].AdaptInt8LutKernelSMemSize();

        dim3 block_size, grid_size;
        block_size.x = cta_size_in_thd;
        block_size.y = 1;
        block_size.z = 1;

        grid_size.x = DivUp(M, tile_m_per_cta);
        //FIXME
        //grid_size.y = DivUp(N_pad, tile_n_per_cta);
        grid_size.y = DivUp(N, tile_n_per_cta);
        grid_size.z = 1; // num_grp * splitk;

        cudaEventRecord(begin, stream);
        for (int i = 0; i < TIMES; i++) {
            if (g_int8_kvec[kid].ktype == CONV_2SPK_F1) {
                FAKE_INT8_CONV_PARAM
                int kLoopNum = DivUp(K_pad, tile_k_per_cta);
                lut_t in_lut, flt_lut;
                (g_int8_kvec[kid].int8_lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_GEMM_FUNC_PARAM);
            }
        }
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, begin, end);

        if (elapsed < minTime) {
            best_kid = kid;
            minTime  = elapsed;
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    algo_param.kid = best_kid;
    return minTime;
#else
    return 0.0;
#endif
}

template <typename T>
ppl::common::RetCode PPLCUDAGemvForwardImpInt8(
    const cudaStream_t &stream,
    const int M,
    const int N,
    const int padN,
    const int K,
    const void *input,
    const void *weight,
    const void *bias,
    void *output,
    const ppl::nn::onnx::GemmParam &param,
    void *temp_buffer,
    const quant_param_t &quant_param,
    const fuse_param_t &fuse_param);

ppl::common::RetCode PPLCUDAGemmForwardImpInt8(
    int device_id,
    const cudaStream_t &stream,
    ppl::nn::cuda::CUDAModule *module,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *weight_shape,
    const void *weight,
    const void *bias,
    const ppl::nn::TensorShape *output_shape,
    void *output,
    const ppl::nn::onnx::GemmParam &param,
    void *temp_buffer,
    const quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    const algo_param_t &algo_param)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020 
    auto type = weight_shape->GetDataType();
    float alpha  = param.alpha;
#ifndef PPLNN_ENABLE_CUDA_JIT
    if (!is_g_int8_kvec_set)
        init_f1_int8_kvec(g_int8_kvec, device_id, type);
#endif
    int pad_size = GetPadSize(type);
    int transA   = param.transA;
    int transB   = param.transB;
    if (!param.transB)
        return ppl::common::RC_UNSUPPORTED;
    int N     = transB ? weight_shape->GetDim(0) : weight_shape->GetDim(1);
    int K     = transB ? weight_shape->GetDim(1) : weight_shape->GetDim(0);
    int N_pad = Align(N, pad_size);
    int K_pad = Align(K, pad_size);
    int M     = transA ? input_shape->GetDim(1) : input_shape->GetDim(0);

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;
    int4 *final_out      = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : (int4 *)output;

    // fuse configs
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if (M == 1) { // TODO, only work for A100, need to diff with T4
        status = PPLCUDAGemvForwardImpInt8<int8_t>(stream,
                                               M,
                                               N,
                                               N_pad,
                                               K_pad,
                                               input,
                                               weight,
                                               bias,
                                               (void *)final_out,
                                               param,
                                               temp_buffer,
                                               quant_param,
                                               fuse_param);
        return status;
    }
    // kernel configs
#ifdef PPLNN_ENABLE_CUDA_JIT
    int tile_m_per_cta  = algo_param.tiles.m_cta;
    int tile_n_per_cta  = algo_param.tiles.n_cta;
    int tile_k_per_cta  = algo_param.tiles.k_cta;
    int cta_size_in_thd = algo_param.tiles.cta_size_in_thd;
    int smem_size       = algo_param.tiles.smem_size;
#else
    int kid             = algo_param.kid;
    int tile_m_per_cta  = g_int8_kvec[kid].tile_m_per_cta;
    int tile_n_per_cta  = g_int8_kvec[kid].tile_n_per_cta;
    int tile_k_per_cta  = g_int8_kvec[kid].tile_k_per_cta;
    int cta_size_in_thd = g_int8_kvec[kid].cta_size_in_thd;
    int smem_size       = g_int8_kvec[kid].smem_size;
#endif
    dim3 block_size, grid_size;

    block_size.x = cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;
    grid_size.x  = DivUp(M, tile_m_per_cta);
    grid_size.y  = DivUp(N_pad, tile_n_per_cta);
    grid_size.z  = 1; // num_grp * splitk;
    int kLoopNum = DivUp(K_pad, tile_k_per_cta);
    lut_t in_lut, flt_lut;

    int has_bias    = (bias ? 1 : 0); // beta != 0.f;
    int4 *input0_tmp = (int4 *)input;
    if (transA == 1) {
        int K_pad_v4 = K_pad / 4;
        dim3 grid(DivUp(K_pad_v4, 32), DivUp(M, 32), 1);
        dim3 block(32, 32, 1);
        int8_matrix_transpose<<<grid, block, 0, stream>>>((int*)temp_buffer, (int*)input, K_pad_v4, M);
        input0_tmp = (int4 *)temp_buffer;
    }
    FAKE_INT8_CONV_PARAM
#ifdef PPLNN_ENABLE_CUDA_JIT
    int in_lut_size  = 0;
    int flt_lut_size = 0;
    void *prelu      = (void *)fuse_param.prelu;
    void *pre_data   = (void *)fuse_param.pre_data;
    void *elt_prelu  = (void *)fuse_param.elt_prelu;
    float in_quant   = alpha * quant_param.in_scale;
    void *flt_quant  = (void *)quant_param.d_flt_scale;
    float out_quant  = quant_param.out_scale;
    float pre_quant  = quant_param.pre_scale;

    void *args[]        = {&input0_tmp, &weight, &final_out, &kLoopNum, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &in_height, &in_width, &batch, &num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &flt_height, &flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &out_height, &out_width, &stride_height, &stride_width,  &pad_height, &pad_width, &hole_height, &hole_width, &has_bias, &bias, &(in_quant), &(flt_quant), &(out_quant), &(pre_quant), &fuse_param.has_activation, &fuse_param.clip_min, &fuse_param.has_clip, &fuse_param.clip_max, &fuse_param.has_prelu, &(prelu), &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &fuse_param.elt_clip_min, &fuse_param.has_elt_clip,  &fuse_param.elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &fuse_param.leaky, &fuse_param.elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
    CUfunction function = module->GetKernelFunc();

    if(smem_size > MAX_STATIC_SMEM_SIZE_PER_CTA)
        cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size);

    CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
#else
        g_int8_kvec[kid].AdaptInt8LutKernelSMemSize();
        (g_int8_kvec[kid].int8_lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(INT8_GEMM_FUNC_PARAM);
#endif
    return status;
#else
    return ppl::common::RC_UNSUPPORTED;
#endif
}

template <typename T>
__device__ __inline__ void fma_v4(const int4 a, const int4 b, int4 &c);

template <>
__device__ __inline__ void fma_v4<int8_t>(const int4 a, const int4 b, int4 &c)
{
#if __CUDA_ARCH__ >= 600
    ((int *)&c)[0] = __dp4a(((int *)&a)[0], ((int *)&b)[0], ((int *)&c)[0]);
    ((int *)&c)[1] = __dp4a(((int *)&a)[1], ((int *)&b)[1], ((int *)&c)[1]);
    ((int *)&c)[2] = __dp4a(((int *)&a)[2], ((int *)&b)[2], ((int *)&c)[2]);
    ((int *)&c)[3] = __dp4a(((int *)&a)[3], ((int *)&b)[3], ((int *)&c)[3]);
#else
#endif
}
template <>
__device__ __inline__ void fma_v4<__half>(const int4 a, const int4 b, int4 &c)
{
#if __CUDA_ARCH__ >= 600
    ((__half2 *)&c)[0] = __hfma2(((__half2 *)&a)[0], ((__half2 *)&b)[0], ((__half2 *)&c)[0]);
    ((__half2 *)&c)[1] = __hfma2(((__half2 *)&a)[1], ((__half2 *)&b)[1], ((__half2 *)&c)[1]);
    ((__half2 *)&c)[2] = __hfma2(((__half2 *)&a)[2], ((__half2 *)&b)[2], ((__half2 *)&c)[2]);
    ((__half2 *)&c)[3] = __hfma2(((__half2 *)&a)[3], ((__half2 *)&b)[3], ((__half2 *)&c)[3]);
#else
#endif
}
template <>
__device__ __inline__ void fma_v4<float>(const int4 a, const int4 b, int4 &c)
{
    ((float *)&c)[0] = ((float *)&a)[0] * ((float *)&b)[0] + ((float *)&c)[0];
    ((float *)&c)[1] = ((float *)&a)[1] * ((float *)&b)[1] + ((float *)&c)[1];
    ((float *)&c)[2] = ((float *)&a)[2] * ((float *)&b)[2] + ((float *)&c)[2];
    ((float *)&c)[3] = ((float *)&a)[3] * ((float *)&b)[3] + ((float *)&c)[3];
}

template <typename T>
__device__ __inline__ int4 add_v4(const int4 a, const int4 b);

template <>
__device__ __inline__ int4 add_v4<__half>(const int4 a, const int4 b)
{
    int4 res = {0, 0, 0, 0};
#if __CUDA_ARCH__ >= 600
    ((__half2 *)&res)[0] = __hadd2(((__half2 *)&a)[0], ((__half2 *)&b)[0]);
    ((__half2 *)&res)[1] = __hadd2(((__half2 *)&a)[1], ((__half2 *)&b)[1]);
    ((__half2 *)&res)[2] = __hadd2(((__half2 *)&a)[2], ((__half2 *)&b)[2]);
    ((__half2 *)&res)[3] = __hadd2(((__half2 *)&a)[3], ((__half2 *)&b)[3]);
#else
#endif
    return res;
}
template <>
__device__ __inline__ int4 add_v4<float>(const int4 a, const int4 b)
{
    int4 res           = {0, 0, 0, 0};
    ((float *)&res)[0] = ((float *)&a)[0] + ((float *)&b)[0];
    ((float *)&res)[1] = ((float *)&a)[1] + ((float *)&b)[1];
    ((float *)&res)[2] = ((float *)&a)[2] + ((float *)&b)[2];
    ((float *)&res)[3] = ((float *)&a)[3] + ((float *)&b)[3];
    return res;
}

template <typename T>
__inline__ __device__ T reduce_v4(int4 data)
{
    T res = (T)0;
    for (int i = 0; i < sizeof(int4) / sizeof(T); i++) {
        res = Math<T, T, T>::add(res, ((T *)&data)[i]);
    }
}

template <typename T>
__device__ __inline__ void activation(const int activation, int4 &v)
{
    T *t_v                        = (T *)&v;
    constexpr int T_NUMS_PER_INT4 = sizeof(int4) / sizeof(T);
    if (activation == 1) {
        for (int i = 0; i < T_NUMS_PER_INT4; i++)
            t_v[i] = Math<T, T, T>::ge(t_v[i], (T)0) ? t_v[i] : (T)0;
    } else {
        for (int i = 0; i < T_NUMS_PER_INT4; i++) {
            T tmp  = expf(t_v[i]);
            t_v[i] = tmp * __frcp_rn(tmp + (T)1);
        }
    }
}
template <>
__device__ __inline__ void activation<__half>(const int activation, int4 &v)
{
#if __CUDA_ARCH__ >= 600
    __half2 *h2_v = (__half2 *)&v;
    int *int_v    = (int *)&v;
    if (activation == 1) {
        for (int i = 0; i < 4; i++)
            int_v[i] = __vmaxs2(int_v[i], 0);
    } else {
        __half2 one = {(__half)1.f, (__half)1.f};
        for (int i = 0; i < 4; i++) {
            __half2 tmp = h2exp(h2_v[i]);
            h2_v[i]     = __hmul2(tmp, h2rcp(__hadd2(one, tmp))); // __h2div(tmp, __hadd2(one, tmp));
        }
    }
#else
#endif
}
template <typename T>
__device__ __inline__ void clip(int4 &v, float clip_min, float clip_max)
{
    T *t_v                        = (T *)&v;
    constexpr int T_NUMS_PER_INT4 = sizeof(int4) / sizeof(T);
    for (int i = 0; i < T_NUMS_PER_INT4; i++) {
        t_v[i] = Math<T, T, T>::ge(t_v[i], (T)clip_min) ? t_v[i] : (T)clip_min;
        t_v[i] = Math<T, T, T>::le(t_v[i], (T)clip_max) ? t_v[i] : (T)clip_max;
    }
}

// partition workload according to input matrix
// matrix: NxK
// N should not be paded in matrix:(N, padK), but output is (1, N_int4_pad)
//  N: pad int4
//  K: pad int4
//  layout and fuse pattern  consistent with gemm
// BLK_TILE_N: min:8
// THD_TILE_N_V4: num of int
template <typename T, int BLK_TILE_N, int THD_TILE_N_V4, int BLK_SIZE>
__global__ void int8_gemv(void *output,
                     const void *vec,
                     const void *matrix,
                     const void *bias,
                     const int padK,
                     const int N,
                     const int padOutN,//pad int4, 16
                     const fuse_param_t fuse_param,
                     const quant_param_t quant_param)
{
    // blk conofig
    // one int4 per thd along K
    constexpr int T_NUMS_PER_INT  = sizeof(int)  / sizeof(T);
    constexpr int T_NUMS_PER_INT4 = sizeof(int4) / sizeof(T);
    constexpr int DT_NUMS_PER_INT4= sizeof(int4) / sizeof(int);
    constexpr int BLK_TILE_N_V4   = BLK_TILE_N / T_NUMS_PER_INT;
    constexpr int THD_TILE_N      = THD_TILE_N_V4 * T_NUMS_PER_INT;
    constexpr int BLK_SIZE_Y      = BLK_TILE_N_V4 / THD_TILE_N_V4;
    constexpr int BLK_SIZE_X      = BLK_SIZE / BLK_SIZE_Y;
    constexpr int BLK_TILE_K      = BLK_SIZE_X;
    bool in_n_range[THD_TILE_N];
#pragma unroll
    for (int i = 0; i < THD_TILE_N; i++) {
        in_n_range[i] = blockIdx.x * BLK_TILE_N + threadIdx.y + i * BLK_SIZE_Y < N;
    }
    int pad_k_v4                  = padK / T_NUMS_PER_INT4;// 16 int8
    int pad_out_n_v4              = padOutN / DT_NUMS_PER_INT4;// 16 int8
    int n_id                      = blockIdx.x * BLK_TILE_N + threadIdx.y * T_NUMS_PER_INT;
    //dequant
    float4 flt_scale_v4[THD_TILE_N_V4];
    float *flt_scale = (float*)flt_scale_v4;
    for (int i = 0; i < THD_TILE_N_V4; i++) {
        flt_scale_v4[i] = ((float4*)quant_param.d_flt_scale)[blockIdx.x * BLK_TILE_N_V4 + i * BLK_SIZE_Y + threadIdx.y];
    }

    int64_t b_base_v4    = (int64_t)n_id * pad_k_v4;
    int4 *matrix_base_v4 = (int4 *)matrix + b_base_v4;
    int4 reg_c[THD_TILE_N];
    int4 reg_b[THD_TILE_N];
    int4 reg_a;
    int4 zero       = {0, 0, 0, 0};
    int c[THD_TILE_N] = {int(0)};
#pragma unroll
    for (int i = 0; i < THD_TILE_N; i++)
        c[i] = (int)0;
#pragma unroll
    for (int i = 0; i < THD_TILE_N; i++) {
        reg_c[i] = zero;
    }

    // ld global VxM
    // each tid.y ld thd_tile_n/4 * 4(int8) along n, get thd_tile_n (int32) result
#pragma unroll
    for (int k = 0; k < DivUp(pad_k_v4, BLK_TILE_K); k++) {
        int64_t off   = k * BLK_TILE_K + threadIdx.x;
        bool in_range = off < pad_k_v4;
        reg_a         = in_range ? ((int4 *)vec)[off] : zero;
#pragma unroll
        for (int i = 0; i < THD_TILE_N_V4; i++) {
#pragma unroll
            for (int j = 0; j < T_NUMS_PER_INT; j++) {
                reg_b[i * T_NUMS_PER_INT + j] = in_n_range[i * T_NUMS_PER_INT + j] && in_range ?
                                       matrix_base_v4[(i * T_NUMS_PER_INT * BLK_SIZE_Y + j) * pad_k_v4 + off]
                                       : zero;
                fma_v4<T>(reg_a, reg_b[i * T_NUMS_PER_INT + j], reg_c[i * T_NUMS_PER_INT + j]);
            }
        }
    }
    // int4 reduce to half
#pragma unroll
    for (int i = 0; i < THD_TILE_N; i++) {
#pragma unroll
        for (int n = 0; n < DT_NUMS_PER_INT4; n++) {
            c[i] = ((int *)reg_c)[i * DT_NUMS_PER_INT4 + n] + c[i];
        }
    }
    __shared__ int smem[BLK_SIZE_X * BLK_TILE_N];

    int reduce_off            = (threadIdx.y * THD_TILE_N) * BLK_SIZE_X + threadIdx.x;
    constexpr int REDUCE_SIZE = BLK_SIZE_X;
    if (REDUCE_SIZE >= 64) {
#pragma unroll
        for (int i = 0; i < THD_TILE_N; i++) {
            smem[reduce_off + i * BLK_SIZE_X] = c[i];
        }
        __syncthreads();
    }

    // reduce
    if (REDUCE_SIZE >= 1024) {
        if (threadIdx.x < 512)
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                smem[reduce_off + i * BLK_SIZE_X] = smem[reduce_off + i * BLK_SIZE_X] +
                                              smem[512 + reduce_off + i * BLK_SIZE_X];
        __syncthreads();
    }
    if (REDUCE_SIZE >= 512) {
        if (threadIdx.x < 256)
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                smem[reduce_off + i * BLK_SIZE_X] = smem[reduce_off + i * BLK_SIZE_X] +
                                              smem[256 + reduce_off + i * BLK_SIZE_X];
        __syncthreads();
    }
    if (REDUCE_SIZE >= 256) {
        if (threadIdx.x < 128)
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                smem[reduce_off + i * BLK_SIZE_X] = smem[reduce_off + i * BLK_SIZE_X] +
                                              smem[128 + reduce_off + i * BLK_SIZE_X];
        __syncthreads();
    }
    if (REDUCE_SIZE >= 128) {
        if (threadIdx.x < 64)
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                smem[reduce_off + i * BLK_SIZE_X] = smem[reduce_off + i * BLK_SIZE_X] +
                                               smem[64 + reduce_off + i * BLK_SIZE_X];
        __syncthreads();
    }

    unsigned FULL_MASK = __activemask();
    if (REDUCE_SIZE >= 64) {
        if (threadIdx.x < 32) {
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                c[i] = smem[reduce_off + i * BLK_SIZE_X] + smem[reduce_off + i * BLK_SIZE_X + 32];
        }
    }
    if (threadIdx.x < 32) {
        if (REDUCE_SIZE >= 32) {
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                c[i] = c[i] + __shfl_down_sync(FULL_MASK, c[i], 16);
        }
        if (REDUCE_SIZE >= 16) {
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                c[i] = c[i] + __shfl_down_sync(FULL_MASK, c[i], 8);
        }
        if (REDUCE_SIZE >= 8) {
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                c[i] = c[i] + __shfl_down_sync(FULL_MASK, c[i], 4);
        }
        if (REDUCE_SIZE >= 4) {
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                c[i] = c[i] + __shfl_down_sync(FULL_MASK, c[i], 2);
        }
        if (REDUCE_SIZE >= 2) {
#pragma unroll
            for (int i = 0; i < THD_TILE_N; i++)
                c[i] = c[i] + __shfl_down_sync(FULL_MASK, c[i], 1);
        }
    }

    //FIXME ld can be placed in the beginning
    //dequant
    float *fc = (float*)c;
    for (int i = 0; i < THD_TILE_N; i++){
        fc[i] = quant_param.in_scale * flt_scale[i] * c[i];
    }

    // shared shuffle
    int4 *smem_v4 = (int4 *)smem;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < THD_TILE_N_V4; i++) {
            smem_v4[i * BLK_SIZE_Y + threadIdx.y] = ((int4 *)fc)[i];
        }
    }
    __syncthreads();

    int tid = threadIdx.y * BLK_SIZE_X + threadIdx.x;
    for (int thd_off = tid; thd_off < BLK_TILE_N_V4; thd_off += BLK_SIZE) {
        int out_off          = blockIdx.x * BLK_TILE_N_V4 + thd_off;
        bool in_output_range = out_off < pad_out_n_v4;

        if (in_output_range) {
            int4 bias_data = bias != NULL ? ((int4 *)bias)[out_off] : zero;
            // TODO add bias
            int4 out       = add_v4<float>(smem_v4[thd_off], bias_data);

            // fuse
            if (fuse_param.has_activation)
                activation<T>(fuse_param.has_activation, out);
            if (fuse_param.has_clip)
                clip<T>(out, fuse_param.clip_min, fuse_param.clip_max);
            int concatV4_off = 0;
            if (fuse_param.has_concat) {
                int concat_offset_v4 = fuse_param.concat_offset / T_NUMS_PER_INT4;
                int concat_stride_v4 = fuse_param.concat_stride / T_NUMS_PER_INT4;
                concatV4_off         = concat_offset_v4 + blockIdx.y * concat_stride_v4;
                out_off += concatV4_off;
            }

            //quant
#define quantOutData(_C, _fC, _outInt8Scale){ \
	   _C.x = __float2int_rn(_fC[0]*_outInt8Scale); \
	   _C.y = __float2int_rn(_fC[1]*_outInt8Scale); \
	   _C.z = __float2int_rn(_fC[2]*_outInt8Scale); \
	   _C.w = __float2int_rn(_fC[3]*_outInt8Scale); \
}
#define packchar4(_outData, x, y, z, w){ \
    if (x>127)	x = 127;                 \
    if (x<-128) x = -128;                \
    if (y>127)	y = 127;                 \
    if (y<-128) y = -128;                \
    if (z>127)	z = 127;                 \
    if (z<-128) z = -128;                \
    if (w>127)	w = 127;                 \
    if (w<-128) w = -128;                \
    x = (0xffu & (int8_t)x);             \
    y = (0xffu & (int8_t)y) << 8;        \
    z = (0xffu & (int8_t)z) << 16;       \
    w = (0xffu & (int8_t)w) << 24;         \
    _outData = w | z | y | x;/*(x,y,z,w)*/\
}
            quantOutData(out, ((float*)&out), quant_param.out_scale);
            int out_data;
            packchar4(out_data, out.x, out.y, out.z, out.w);
#undef quantOutData
#undef packchar4

            ((int *)output)[out_off] = out_data;
        }
    }
}

template <typename T>
ppl::common::RetCode PPLCUDAGemvForwardImpInt8(
    const cudaStream_t &stream,
    const int M,
    const int N,
    const int padN,
    const int padK,
    const void *input,
    const void *weight,
    const void *bias,
    void *output,
    const ppl::nn::onnx::GemmParam &param,
    void *temp_buffer,
    const quant_param_t &quant_param,
    const fuse_param_t &fuse_param)
{
    if (!param.transB)
        return ppl::common::RC_UNSUPPORTED;

    constexpr int ELEM_NUM_PR_ST = sizeof(int) / sizeof(T);
    constexpr int expect_blocks  = 64;
    int n_v4                     = padN / ELEM_NUM_PR_ST;
    int blk_tile_n_v4            = DivUp(n_v4, expect_blocks / M);
#define LAUNCH_KERNEL()                                                                                                                  \
    {                                                                                                                                    \
        constexpr int BLK_TILE_N = BLK_SIZE_Y * THD_TILE_N_V4 * ELEM_NUM_PR_ST;                                                          \
        constexpr int BLK_SIZE   = BLK_SIZE_Y * BLK_SIZE_X;                                                                              \
        dim3 grid;                                                                                                                       \
        grid.x       = DivUp(padN, BLK_TILE_N);                                                                                          \
        grid.y       = 1;                                                                                                                \
        grid.z       = 1;                                                                                                                \
        dim3 threads = dim3(BLK_SIZE_X, BLK_SIZE_Y, 1);                                                                                  \
        int8_gemv<T, BLK_TILE_N, THD_TILE_N_V4, BLK_SIZE><<<grid, threads, 0, stream>>>(output, input, weight, bias, padK, N, padN, fuse_param, quant_param); \
    }
#define CONFIG_KERNEL(_blk_tile_n_v4)                  \
    {                                                  \
        if (BLK_SIZE_X <= 32 && _blk_tile_n_v4 >= 16) { \
            constexpr int THD_TILE_N_V4 = 4;           \
            constexpr int BLK_SIZE_Y    = 4;           \
            LAUNCH_KERNEL();                           \
        } else if (_blk_tile_n_v4 >= 8) {               \
            constexpr int THD_TILE_N_V4 = 4;           \
            constexpr int BLK_SIZE_Y    = 2;           \
            LAUNCH_KERNEL();                           \
        } else if (_blk_tile_n_v4 >= 4) {               \
            constexpr int THD_TILE_N_V4 = 2;           \
            constexpr int BLK_SIZE_Y    = 2;           \
            LAUNCH_KERNEL();                           \
        } else if (_blk_tile_n_v4 >= 2) {               \
            constexpr int THD_TILE_N_V4 = 2;           \
            constexpr int BLK_SIZE_Y    = 1;           \
            LAUNCH_KERNEL();                           \
        } else {                                       \
            constexpr int THD_TILE_N_V4 = 1;           \
            constexpr int BLK_SIZE_Y    = 1;           \
            LAUNCH_KERNEL();                           \
        }                                              \
    }
    if (padK >= 8196) {
        constexpr int BLK_SIZE_X = 128;
        CONFIG_KERNEL(blk_tile_n_v4);
    } else if (padK >= 1024) {
        constexpr int BLK_SIZE_X = 64;
        CONFIG_KERNEL(blk_tile_n_v4);
    } else {
        constexpr int BLK_SIZE_X = 32;
        CONFIG_KERNEL(blk_tile_n_v4);
    }

    return ppl::common::RC_SUCCESS;
}
