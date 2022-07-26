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

#include "cudakernel/gemm/bgemm.h"
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

// defined in gemm.cu
extern std::vector<kernel_info_t> g_fp16_kvec;
extern bool is_g_fp16_kvec_set;

#define FAKE_CONV_PARAM              \
    int in_hw               = 1;     \
    int out_hw              = 1;     \
    int flt_hw              = 1;     \
    int splitk              = 1;     \
    int in_height           = 1;     \
    int in_width            = 1;     \
    int conv_batch          = M;     \
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

#define GEMM_FUNC_PARAM                                               \
        input0_tmp,                                                   \
        tmp_weight,                                                   \
        final_out,                                                    \
        kLoopNum,                                                     \
        in_lut, 0,                                                    \
        flt_lut, 0,                                                   \
        in_hw, out_hw,                                                \
        flt_hw, splitk,                                               \
        in_height, in_width,                                          \
        conv_batch, num_grp,                                          \
        num_chl_per_grp, num_chl_per_grp_pad,                         \
        flt_height, flt_width,                                        \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        out_height, out_width,                                        \
        stride_height, stride_width,                                  \
        pad_height, pad_width,                                        \
        hole_height, hole_width,                                      \
        has_bias, (int4 *)bias,                                       \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        (__half)fuse_param.leaky, (__half)fuse_param.elt_leaky,       \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

extern void init_f1_kvec(std::vector<kernel_info_t> &g_fp16_kvec, int device_id, ppl::common::datatype_t type);

uint64_t PPLBgemmCUDAGetBufSize(
    const ppl::nn::TensorShape *input_shape,
    int transA)
{
    return 0;
}

template <typename T>
__global__ void matrix_transpose(
    T *output,
    T *input,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width)
{
    __shared__ T shared[32][33];

    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int in_h = bx * 32;
    int in_w = by * 32 + tid;
    int out_h = by * 32;
    int out_w = bx * 32 + tid;

    int in_idx = bz * in_height * in_width + in_h * in_width + in_w;
    int out_idx = bz * out_height * out_width + out_h * out_width + out_w;

    if (in_w < in_width) {
        for (int i = 0; i < 32; i++){
            if (in_h + i < in_height)
                shared[i][tid] = input[in_idx + i * in_width];
        }
    }
    __syncthreads();
    T reg_zero = 0;
    if (out_w < out_width) {
        for (int i = 0; i < 32; i++) {
            if (out_h + i < out_height) {
                output[out_idx + i * out_width] = (out_w < in_height && (out_h + i) < in_width) ?
                                                shared[tid][i] : reg_zero;
            }
        }
    }
}

ppl::common::RetCode PPLCUDABgemmModifyWeights(
    const cudaStream_t &stream,
    ppl::nn::TensorShape *weight_shape,
    void *weight,
    void *tmp_weight, // if need pad transpose
    const ppl::nn::onnx::GemmParam *param)
{
    //int transB   = param->transB;
    //float alpha  = param->alpha;
    auto type    = weight_shape->GetDataType();
    int pad_size = GetPadSize(type);

    int dim_count = weight_shape->GetDimCount();
    int batch = 1;
    for (int i = 0; i < dim_count-2; i++){
        batch *= weight_shape->GetDim(i);
    }
    const int dim0 = weight_shape->GetDim(dim_count - 2); // original shape 
    const int dim1 = weight_shape->GetDim(dim_count - 1);
    const int dim0_pad = Align(dim0, pad_size);

#define TRANSWEIGHT(Type)                                                                                      \
    matrix_transpose<Type><<<grid, block, 0, stream>>>((Type *)tmp_weight, (Type *)weight, dim0, dim1, dim1, dim0_pad); \

        dim3 grid(DivUp(dim0_pad, 32), DivUp(dim1, 32), batch);
        dim3 block(32, 1, 1);
        switch (type) {
            case ppl::common::DATATYPE_FLOAT32: {
                TRANSWEIGHT(float)
                break;
            }
            case ppl::common::DATATYPE_FLOAT16: {
                TRANSWEIGHT(__half)

                break;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
#undef TRANSWEIGHT

    return ppl::common::RC_SUCCESS;
}

template<typename T>
__global__ void pad_matrix(T *input, T *output, uint64_t outer, int ori_inner, int align_inner){
    uint64_t out_off = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_o_range = out_off < outer * (uint64_t)align_inner;
    int inner_id = out_off % align_inner;
    uint64_t outer_id = out_off / align_inner;
    bool in_i_range = outer_id < outer && inner_id < ori_inner;
    int in_off = outer_id * ori_inner + inner_id;
    T value = in_i_range ? input[in_off] : (T)0;

    if (in_o_range)    output[out_off] = value;
}
ppl::common::RetCode PPLCUDABgemmPadInput(
    const cudaStream_t &stream,
    ppl::nn::TensorShape *input_shape,
    void *input,
    void *tmp_input, // if need transpose
    const ppl::nn::onnx::GemmParam *param)
{
    auto type    = input_shape->GetDataType();
    int pad_size = GetPadSize(type);

    int dim_count0 = input_shape->GetDimCount();
    uint64_t batch = 1;
    for (int i = 0; i < dim_count0-2; i++){
        batch *= input_shape->GetDim(i);
    }
    const int dim0 = input_shape->GetDim(dim_count0 - 2); // original shape 
    const int dim1 = input_shape->GetDim(dim_count0 - 1);
    const int dim1_pad = Align(dim1, pad_size);
    dim3 block(512,1,1);
    dim3 grid(1,1,1);
    uint64_t size = batch * dim0 * dim1_pad;
    grid.x = DivUp(size, block.x);
    //grid.y = dim0 * batch;
    switch (type) {
        case ppl::common::DATATYPE_FLOAT32: {
            pad_matrix<<<grid, block, 0, stream>>>((float*)input, (float*)tmp_input,
                                                   batch*dim0, dim1, dim1_pad);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16: {
            pad_matrix<<<grid, block, 0, stream>>>((__half*)input, (__half*)tmp_input,
                                                   batch*dim0, dim1, dim1_pad);
            break;
        }
        default:
                return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
template<typename T>
__global__ void matrix_rm_pad(T *input, T *output, uint64_t outer, int ori_inner, int align_inner){
    uint64_t out_off = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_o_range = out_off < outer * ori_inner;
    int inner_id = out_off % ori_inner;
    int outer_id = out_off / ori_inner;
    int in_off = outer_id * align_inner + inner_id;

    if (in_o_range)    output[out_off] = input[in_off];
}

ppl::common::RetCode PPLCUDABgemmCvtOutput(
    const cudaStream_t &stream,
    ppl::nn::TensorShape *output_shape,
    void *output,
    void *tmp_output)
{
    auto type = output_shape->GetDataType();
    int pad_size = GetPadSize(type);

    int dim_count = output_shape->GetDimCount();
    uint64_t batch = 1;
    for (int i = 0; i < dim_count-2; i++){
        batch *= output_shape->GetDim(i);
    }
    const int dim0 = output_shape->GetDim(dim_count - 2); // original shape 
    const int dim1 = output_shape->GetDim(dim_count - 1);
    const int dim1_pad = Align(dim1, pad_size);
    dim3 block(512,1,1);
    dim3 grid(1,1,1);
    uint64_t size = batch * dim0 * dim1;
    grid.x = DivUp(size, block.x);
    //grid.y = dim0 * batch;
    switch (type) {
        case ppl::common::DATATYPE_FLOAT32: {
            matrix_rm_pad<<<grid, block, 0, stream>>>((float*)tmp_output, (float*)output,
                                                   batch*dim0, dim1, dim1_pad);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16: {
            matrix_rm_pad<<<grid, block, 0, stream>>>((__half*)tmp_output, (__half*)output,
                                                   batch*dim0, dim1, dim1_pad);
            break;
        }
        default:
                return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

#define MAX_KERNEL_SIZE (1 + 12 + 30)

__inline__ std::string ToString(int v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

double PPLCUDABgemmJITSelectKernel(
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
    fuse_param_t &fuse_param,
    algo_param_t &algo_param,
    uint64_t workspace)
{
    double elapsed = 0.0f;
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::vector<std::string> knames;
    std::vector<algo_param_t> params;
    std::string sources = "";

    GetFp16ConvKernelNominees(device_id, type, conv_param, knames, params, sources);

    int index = 0;
    std::vector<const char *> compile_params;
    elapsed = AlgoForwardTime(device_id, stream, knames, sources, index, compile_params, device_id, true, type, (int4 *)input, (int4 *)weight, (int4 *)output, (int4 *)bias, (int4 *)temp_buffer, params, conv_param, fuse_param, workspace);

    algo_param = params[index];
#endif
    return elapsed;
}

double PPLCUDABgemmSelectKernel(
    int device_id,
    const cudaStream_t &stream,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *weight_shape,
    void *weight,
    const ppl::nn::TensorShape *output_shape,
    void *output,
    void *temp_buffer,
    const ppl::nn::onnx::GemmParam &param,
    const fuse_param_t &fuse_param,
    algo_param_t &algo_param)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    auto type = weight_shape->GetDataType();
    if (!is_g_fp16_kvec_set)
        init_f1_kvec(g_fp16_kvec, device_id, type);

    int pad_size = GetPadSize(type);

    // FIXME use non-paded N in conv1x1 for input
    auto dim_count0 = input_shape->GetDimCount();
    auto dim_count1 = weight_shape->GetDimCount();
    if (input_shape->GetDim(dim_count0-1) != weight_shape->GetDim(dim_count1-2))
        return FLT_MAX;
    //FIXME Dim is 64 bit?
    int m_id = dim_count0 - 2;
    while(m_id && input_shape->GetDim(m_id)==1)    m_id--;
    int M = input_shape->GetDim(m_id);
    uint64_t batch = 1;
    for (int i = 0; i < m_id; i++){
        batch *= input_shape->GetDim(i);
    }
    if (dim_count1 == 2){
        M *= batch;
        batch = 1;
    }
    int K_pad = input_shape->GetDim(dim_count0-1);
    int N     = weight_shape->GetDim(dim_count1-1);
    int N_pad = Align(N, pad_size);

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;
    int4 *final_out      = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : (int4 *)output;

    // fuse configs
    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    bool has_bias        = false; // beta != 0.f;
    half *bias = NULL;
    int4 *tmp_weight = reinterpret_cast<int4*>(weight);

    float minTime = FLT_MAX;
    int best_kid  = -1;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // transpose
    int4 *input0_tmp = (int4 *)input;

    for (unsigned int kid = 0; kid < g_fp16_kvec.size(); kid++) {
        int tile_m_per_cta = g_fp16_kvec[kid].tile_m_per_cta;
        int tile_n_per_cta = g_fp16_kvec[kid].tile_n_per_cta;
        int tile_k_per_cta = g_fp16_kvec[kid].tile_k_per_cta;

        int cta_size_in_thd = g_fp16_kvec[kid].cta_size_in_thd;
        int smem_size       = g_fp16_kvec[kid].smem_size;

        if (!g_fp16_kvec[kid].CheckSMemSizeFeasible(device_prop))
                continue;

        if (!g_fp16_kvec[kid].CheckGpuArchFeasible(device_prop))
                continue;

        g_fp16_kvec[kid].AdaptLutKernelSMemSize();

        dim3 block_size, grid_size;
        block_size.x = cta_size_in_thd;
        block_size.y = 1;
        block_size.z = 1;

        grid_size.x = DivUp(M, tile_m_per_cta);
        grid_size.y = DivUp(N_pad, tile_n_per_cta);
        grid_size.z = 1;

        cudaEventRecord(begin, stream);
        for (int i = 0; i < TIMES; i++) {
            if (g_fp16_kvec[kid].ktype == CONV_2SPK_F1) {
                FAKE_CONV_PARAM
                int kLoopNum = DivUp(K_pad, tile_k_per_cta);
                lut_t in_lut, flt_lut;
                while (batch > 65535) {
                    grid_size.z = 65535;
                    batch -= 65535;
                    (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(GEMM_FUNC_PARAM);
                    input0_tmp += (uint64_t)65535 * M * K_pad / pad_size;// int4
                    tmp_weight += (uint64_t)65535 * N * K_pad / pad_size;// void
                    final_out += (uint64_t)65535 * M * N_pad / pad_size;// int4
                }
                if (batch > 0){
                    grid_size.z = batch;
                    (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(GEMM_FUNC_PARAM);
                }
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

// (B, M, K_pad) * ((B,)N, K_pad) = (B, M, N_pad)
ppl::common::RetCode PPLCUDABgemmForwardImp(
    int device_id,
    const cudaStream_t &stream,
    ppl::nn::cuda::CUDAModule *module,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *weight_shape,
    void *weight,
    const ppl::nn::TensorShape *output_shape,
    void *output,
    const ppl::nn::onnx::GemmParam &param,
    void *temp_buffer,
    fuse_param_t &fuse_param,
    const algo_param_t &algo_param)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020
    auto type = weight_shape->GetDataType();
#ifndef PPLNN_ENABLE_CUDA_JIT
    if (!is_g_fp16_kvec_set)
        init_f1_kvec(g_fp16_kvec, device_id, type);
#endif
    int pad_size = GetPadSize(type);
    //int transA   = param.transA;
    //int transB   = param.transB;
    //if (!param.transB)
    //    return ppl::common::RC_UNSUPPORTED;
    //int N     = transB ? weight_shape->GetDim(0) : weight_shape->GetDim(1);
    //int K     = transB ? weight_shape->GetDim(1) : weight_shape->GetDim(0);
    //int N_pad = Align(N, pad_size);
    //int K_pad = Align(K, pad_size);
    //int M     = transA ? input_shape->GetDim(1) : input_shape->GetDim(0);

    //FIXME Dim is 64 bit?
    //int M     = input_shape->GetDim(dim_count0 - 2);
    auto dim_count0 = input_shape->GetDimCount();
    auto dim_count1 = weight_shape->GetDimCount();
    int m_id = dim_count0 - 2;
    while(m_id && input_shape->GetDim(m_id)==1)    m_id--;
    int M = input_shape->GetDim(m_id);
    uint64_t batch = 1;
    for (int i = 0; i < m_id; i++){
        batch *= input_shape->GetDim(i);
    }
    if (dim_count1 == 2){
        M *= batch;
        batch = 1;
    }
    int K     = input_shape->GetDim(dim_count0- 1);
    int K_pad = Align(K, pad_size);
    int N     = weight_shape->GetDim(dim_count1 - 1);
    int N_pad = Align(N, pad_size);


    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;
    int4 *final_out      = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : (int4 *)output;

    // fuse configs
    __half2 clip_min            = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max            = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min        = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max        = __float2half2_rn(fuse_param.elt_clip_max);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    half *bias = NULL;

    // kernel configs
#ifdef PPLNN_ENABLE_CUDA_JIT
    int tile_m_per_cta  = algo_param.tiles.m_cta;
    int tile_n_per_cta  = algo_param.tiles.n_cta;
    int tile_k_per_cta  = algo_param.tiles.k_cta;
    int cta_size_in_thd = algo_param.tiles.cta_size_in_thd;
#else
    int kid             = algo_param.kid;
    int tile_m_per_cta  = g_fp16_kvec[kid].tile_m_per_cta;
    int tile_n_per_cta  = g_fp16_kvec[kid].tile_n_per_cta;
    int tile_k_per_cta  = g_fp16_kvec[kid].tile_k_per_cta;
    int cta_size_in_thd = g_fp16_kvec[kid].cta_size_in_thd;
    int smem_size       = g_fp16_kvec[kid].smem_size;
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

    bool has_bias    = bias; // beta != 0.f;
    int4 *input0_tmp = (int4 *)input;

    FAKE_CONV_PARAM
#ifdef PPLNN_ENABLE_CUDA_JIT
    int in_lut_size  = 0;
    int flt_lut_size = 0;
    void *prelu      = (void *)fuse_param.prelu;
    void *pre_data   = (void *)fuse_param.pre_data;
    void *elt_prelu  = (void *)fuse_param.elt_prelu;
    half leaky       = fuse_param.leaky;
    half elt_leaky   = fuse_param.elt_leaky;

    void *args[]        = {&input0_tmp, &weight, &final_out, &kLoopNum, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &in_height, &in_width, &conv_batch, &num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &flt_height, &flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &out_height, &out_width, &stride_height, &stride_width, &pad_height, &pad_width, &hole_height, &hole_width, &has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
    CUfunction function = module->GetKernelFunc();
    grid_size.z = batch;
    CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
#else
    int4 *tmp_weight = reinterpret_cast<int4*>(weight);
    while (batch > 65535) {
        grid_size.z = 65535;
        batch -= 65535;
        g_fp16_kvec[kid].AdaptLutKernelSMemSize();
        (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(GEMM_FUNC_PARAM);
        input0_tmp += (uint64_t)65535 * M * K_pad / pad_size;// int4
        tmp_weight += (uint64_t)65535 * N * K_pad / pad_size;// void
        final_out += (uint64_t)65535 * M * N_pad / pad_size;// int4
    }
    if (batch > 0){
        grid_size.z = batch;
        g_fp16_kvec[kid].AdaptLutKernelSMemSize();
        (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(GEMM_FUNC_PARAM);
    }
#endif
    return status;
#else
    return ppl::common::RC_UNSUPPORTED;
#endif
}
