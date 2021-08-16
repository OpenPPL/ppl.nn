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

#include <cuda_fp16.h>
#include <float.h>

#include "kernel_type.h"
#include "conv_common.h"

#define TIMES 4

static std::vector<kernel_info_t> g_kvec;
static bool is_g_kvec_set = false;

#define FAKE_CONV_PARAM \
    const int in_hw = 1;           const int out_hw = 1;                    \
    const int flt_hw = 1;          const int splitk = 1;                    \
    const int in_height = 1;       const int in_width = 1;                  \
    const int batch = M;           const int num_grp = 1;                   \
    const int num_chl_per_grp = 0; const int num_chl_per_grp_pad = K_pad;   \
    const int flt_height = 1;      const int flt_width = 1;                 \
    const int num_flt_per_grp = 0; const int num_flt_per_grp_pad = N_pad;   \
    const int out_height = 1;      const int out_width = 1;                 \
    const int stride_height = 1;   const int stride_width = 1;              \
    const int pad_height = 0;      const int pad_width = 0;                 \
    const int hole_height = 1;     const int hole_width = 1; 

#define GEMM_FUNC_PARAM \
    input0_tmp,                                                             \
    (int4*)weight,                                                          \
    final_out,                                                              \
    kLoopNum,                                                               \
    in_lut,                        0,                                       \
    flt_lut,                       0,                                       \
    in_hw,                         out_hw,                                  \
    flt_hw,                        splitk,                                  \
    in_height,                     in_width,                                \
    batch,                         num_grp,                                 \
    num_chl_per_grp,               num_chl_per_grp_pad,                     \
    flt_height,                    flt_width,                               \
    num_flt_per_grp,               num_flt_per_grp_pad,                     \
    out_height,                    out_width,                               \
    stride_height,                 stride_width,                            \
    pad_height,                    pad_width,                               \
    hole_height,                   hole_width,                              \
    has_bias,                      (int4*)bias,                             \
    fuse_param.has_activation,     clip_min,                                \
    fuse_param.has_clip,           clip_max,                                \
    fuse_param.has_prelu,          (const void *) fuse_param.prelu,       \
    fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,      \
    fuse_param.has_elt_activation, elt_clip_min,                            \
    fuse_param.has_elt_clip,       elt_clip_max,                            \
    fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,   \
    (__half)fuse_param.leaky,      (__half)fuse_param.elt_leaky,            \
    fuse_param.has_concat,         concat_offset_v8,                        \
    concat_stride_v8

void init_f1_kvec(std::vector<kernel_info_t> &g_kvec, ppl::common::datatype_t type)
{
    if ( type == ppl::common::DATATYPE_FLOAT32 )
    {
        printf("fp32 unsupported in %s\n", __FUNCTION__);
    }
    else if ( type == ppl::common::DATATYPE_FLOAT16 )
    {
        Initialize2spkConvF1KernelContainer(g_kvec);
    }
    else 
    { printf("type unsupported\n"); }

    is_g_kvec_set = true;
}

uint64_t PPLGemmCUDAGetBufSize(
    const ppl::nn::TensorShape* input_shape,
    int transA)
{
    auto type = input_shape->GetDataType();
    int type_size = ppl::common::GetSizeOfDataType(type);

    if(transA){
        int pad_size = GetPadSize(type); // ldg 128 bytes
        int K     = input_shape->GetDim(0);
        int M     = input_shape->GetDim(1);
        int K_pad = Align(K, pad_size);
	    return M * K_pad * type_size;
    }
    return 0;
}

unsigned int PPLCUDAGemmGetBiasSize(
    const ppl::common::datatype_t type,
    const int N,
    const bool is_scalar)
{
    if(!is_scalar)    return 0;
    int pad_size = GetPadSize(type); // ldg 128 bytes
    int N_pad    = Align(N, pad_size);
    int type_size = ppl::common::GetSizeOfDataType(type);
    return N_pad * type_size;
}

//block size: (32,32,1)
template<typename T>
__global__ void matrix_transpose(
    T *output,
    T *input,
    float scale,
    const int in_row,
    const int in_col)
{
    unsigned int in_x  = blockIdx.x*32 + threadIdx.x;
    unsigned int in_y  = blockIdx.y*32 + threadIdx.y;
    unsigned int out_x = blockIdx.y*32 + threadIdx.x;
    unsigned int out_y = blockIdx.x*32 + threadIdx.y;
    bool in_range  = (in_x  <= in_col) && (in_y  <= in_row);
    bool out_range = (out_x <= in_row) && (out_y <= in_col);
    __shared__ T smem[32][33];

    T value = in_range ? input[in_y*in_col + in_x] : (T)0;
    smem[threadIdx.x][threadIdx.y] = value;

    __syncthreads();
    value = smem[threadIdx.y][threadIdx.x];
    float fp_value = (float)value * scale;
    if(out_range) output[out_y*in_row + out_x] = (__half)fp_value;
}

template<typename T>
__global__ void scale(T *input, float scale, unsigned int size){
    unsigned int off = blockIdx.x*512 + threadIdx.x;
    bool in_range = off <= size;
    T value = in_range ? input[off] : (T)0;
    float fp_value = (float)value;
    fp_value = scale * fp_value;
    if (in_range)    input[off] = (T)fp_value;
}
ppl::common::RetCode PPLCUDAGemmModifyWeights(
    const cudaStream_t &stream, 
    ppl::nn::TensorShape* weight_shape, 
    void* weight,
    void* tmp_weight, //if need transpose
    const ppl::nn::common::GemmParam *param)
{
    int transB  = param->transB;
    float alpha = param->alpha;
    auto type   = weight_shape->GetDataType();
    int pad_size   = GetPadSize(type);
    
    const int dim0 = weight_shape->GetDim(0);//assume padded
    const int dim1 = weight_shape->GetDim(1);

    if (!transB) {
#define TRANSWEIGHT(Type) \
    matrix_transpose<Type><<<grid, block, 0, stream>>>                          \
        ((Type*)tmp_weight, (Type*)weight, alpha, dim0, dim1);                  \
    cudaMemcpyAsync((Type*)weight, (Type*)tmp_weight, dim0*dim1*sizeof(Type),   \
        cudaMemcpyDeviceToDevice, stream);

        dim3 grid(DivUp(dim1, 32), DivUp(dim0, 32), 1);
        dim3 block(32, 32, 1);
	    weight_shape->SetDim(0, dim1);
        weight_shape->SetDim(1, dim0);
        switch(type){
            case ppl::common::DATATYPE_FLOAT32 : {
                TRANSWEIGHT(float)
                break;
            }
            case ppl::common::DATATYPE_FLOAT16 : {
                TRANSWEIGHT(__half)
                break;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
#undef TRANSWEIGHT
    } else if (alpha != 1.f){
        int grid_size = DivUp(dim0*dim1, 512);
        switch(type){
            case ppl::common::DATATYPE_FLOAT32 : {
                scale<float><<<grid_size, 512, 0, stream>>>((float*)weight, alpha, dim0*dim1);
                break;
            }
            case ppl::common::DATATYPE_FLOAT16 : {
                scale<__half><<<grid_size, 512, 0, stream>>>((__half*)weight, alpha, dim0*dim1);
                break;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
        }        
    }
    return ppl::common::RC_SUCCESS;
}
ppl::common::RetCode PPLCUDAGemmModifyBias(
    const cudaStream_t &stream, 
    const ppl::nn::TensorShape* bias_shape, 
    void* bias, 
    const ppl::nn::common::GemmParam *param)
{
    if (param->bias_term) {
        auto type  = bias_shape->GetDataType();
        int pad_size   = GetPadSize(type);
        float beta = param->beta;
	    int N = bias_shape->GetDim(0);
	    int N_pad = Align(N, pad_size);
        if (type == ppl::common::DATATYPE_FLOAT32) {
	        if (bias_shape->IsScalar())    return ppl::common::RC_UNSUPPORTED;
	        if (beta != 0.f && beta != 1.f){
                int grid_size = DivUp(N_pad, 512);
                scale<float><<<grid_size, 512, 0, stream>>>((float*)bias, beta, N_pad);
	        }
        } else if (type == ppl::common::DATATYPE_FLOAT16) {
            if (bias_shape->IsScalar())    return ppl::common::RC_UNSUPPORTED;
            if (beta != 0.f && beta != 1.f){
                int grid_size = DivUp(N_pad, 512);
                scale<__half><<<grid_size, 512, 0, stream>>>((__half*)bias, beta, N_pad);
            }
        } else{
            return ppl::common::RC_UNSUPPORTED;
        }
    }
    return ppl::common::RC_SUCCESS;
}



int PPLCUDAGemmSelectKernel(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::common::GemmParam &param,
    void* temp_buffer, 
    const fuse_param_t &fuse_param)
{
    auto type = weight_shape->GetDataType();
    if (!is_g_kvec_set) init_f1_kvec(g_kvec, type);

    int pad_size   = GetPadSize(type);
    int transA = param.transA;
    int transB = param.transB;

    int N_pad = transB ? weight_shape->GetDim(0) : weight_shape->GetDim(1);
    int K_pad = transB ? weight_shape->GetDim(1) : weight_shape->GetDim(0);
    int M = transA ? input_shape->GetDim(1) : input_shape->GetDim(0);

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;
    int4 *final_out = fuse_param.has_concat ? (int4*)fuse_param.post_concat : (int4*)output;

    // fuse configs
    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    bool has_bias = param.bias_term;//beta != 0.f;

    float minTime = FLT_MAX;
    int best_kid = -1;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    //transpose
    int4 *input0_tmp = (int4*)input;
    if (transA == 1) { // input is shape of (K, M), we need K as the 1st inner dim
        dim3 grid(DivUp(K_pad, 32), DivUp(M, 32), 1);
        dim3 block(32, 32, 1);
	    if (type == ppl::common::DATATYPE_FLOAT32) {
            matrix_transpose<float><<<grid, block, 0, stream>>>
		                    ((float*)temp_buffer, (float*)input, 1.f, K_pad, M);
        } else if (type == ppl::common::DATATYPE_FLOAT16) {
            matrix_transpose<__half><<<grid, block, 0, stream>>>
		                    ((__half*)temp_buffer, (__half*)input, 1.f, K_pad, M);
	    } else {
            return ppl::common::RC_UNSUPPORTED;
	    }
	    input0_tmp = (int4*)temp_buffer;
    }

    for (unsigned int kid = 0; kid < g_kvec.size(); kid++) {
        int tile_m_per_cta   = g_kvec[kid].tile_m_per_cta;
        int tile_n_per_cta   = g_kvec[kid].tile_n_per_cta;
        int tile_k_per_cta   = g_kvec[kid].tile_k_per_cta;

        int cta_size_in_thd  = g_kvec[kid].cta_size_in_thd;
        dim3 block_size, grid_size;
        block_size.x = cta_size_in_thd;
        block_size.y = 1;
        block_size.z = 1;

        grid_size.x  = DivUp(M, tile_m_per_cta);
        grid_size.y  = DivUp(N_pad, tile_n_per_cta);
        grid_size.z  = 1;//num_grp * splitk;

	    cudaEventRecord(begin, stream);
	    for (int i = 0; i < TIMES; i++) {
            if (g_kvec[kid].ktype == CONV_2SPK_F1) {
                FAKE_CONV_PARAM
	        int kLoopNum = DivUp(K_pad, tile_k_per_cta);
                lut_t in_lut, flt_lut;
                (g_kvec[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(GEMM_FUNC_PARAM);
            }
            else { 
                printf("Error: kernel type error in %s\n", __FUNCTION__); 
            }
        }
	    cudaEventRecord(end, stream);
	    cudaEventSynchronize(end);
	    cudaEventElapsedTime(&elapsed, begin, end);

	    if (elapsed < minTime){
	        best_kid = kid;
	        minTime = elapsed;
	    }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return best_kid;
}

template<typename T>
ppl::common::RetCode PPLCUDAGemvForwardImp(
    const cudaStream_t &stream,
    const int M,
    const int N,
    const int K,
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ppl::nn::common::GemmParam &param,
    void* temp_buffer, 
    const fuse_param_t &fuse_param);


ppl::common::RetCode PPLCUDAGemmForwardImp(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::common::GemmParam &param,
    void* temp_buffer, 
    const fuse_param_t &fuse_param, 
    const int kid)
{
    auto type = weight_shape->GetDataType();
    if ( !is_g_kvec_set ) init_f1_kvec(g_kvec, type);

    int pad_size   = GetPadSize(type);
    int transA   = param.transA;
    int transB   = param.transB;
    if(!param.transB)    return ppl::common::RC_UNSUPPORTED;
    int N     = transB ? weight_shape->GetDim(0) : weight_shape->GetDim(1);
    int K     = transB ? weight_shape->GetDim(1) : weight_shape->GetDim(0);
    int N_pad = Align(N, pad_size);
    int K_pad = Align(K, pad_size);
    int M = transA ? input_shape->GetDim(1) : input_shape->GetDim(0);

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;
    int4 *final_out = fuse_param.has_concat ? (int4*)fuse_param.post_concat : (int4*)output;

    // fuse configs
    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if(M == 1){
        status = PPLCUDAGemvForwardImp<__half>(stream,
                    M, N_pad, K_pad,
                    input, weight, bias,
                    (void*)final_out,
                    param, temp_buffer, fuse_param);
        return status;
    }
    // kernel configs
    int tile_m_per_cta   = g_kvec[kid].tile_m_per_cta;
    int tile_n_per_cta   = g_kvec[kid].tile_n_per_cta;
    int tile_k_per_cta   = g_kvec[kid].tile_k_per_cta;
    int cta_size_in_thd  = g_kvec[kid].cta_size_in_thd;
    dim3 block_size, grid_size;

    block_size.x = cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;
    grid_size.x  = DivUp(M, tile_m_per_cta);
    grid_size.y  = DivUp(N_pad, tile_n_per_cta);
    grid_size.z  = 1;//num_grp * splitk;
    int kLoopNum = DivUp(K_pad, tile_k_per_cta);
    lut_t in_lut, flt_lut;

    bool has_bias = param.bias_term;//beta != 0.f;
    int4 *input0_tmp = (int4*)input;
    if (transA == 1) {
        dim3 grid(DivUp(K_pad, 32), DivUp(M, 32), 1);
        dim3 block(32, 32, 1);
	    if (type == ppl::common::DATATYPE_FLOAT32) {
            matrix_transpose<float><<<grid, block, 0, stream>>>
		            ((float*)temp_buffer, (float*)input, 1.f, K_pad, M);
        } else if (type == ppl::common::DATATYPE_FLOAT16) {
            matrix_transpose<__half><<<grid, block, 0, stream>>>
		            ((__half*)temp_buffer, (__half*)input, 1.f, K_pad, M);
	    } else {
            return ppl::common::RC_UNSUPPORTED;
	    }
	    input0_tmp = (int4*)temp_buffer;
    }
    FAKE_CONV_PARAM

    (g_kvec[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(GEMM_FUNC_PARAM);
    return status;
}




template <typename T>
__device__ __inline__ void fma_v4(const int4 a, const int4 b, int4 &c);

template <>
__device__ __inline__ void fma_v4<__half>(const int4 a, const int4 b, int4 &c){
#if __CUDA_ARCH__ >= 600
    ((__half2*)&c)[0] = __hfma2(((__half2*)&a)[0], ((__half2*)&b)[0], ((__half2*)&c)[0]);
    ((__half2*)&c)[1] = __hfma2(((__half2*)&a)[1], ((__half2*)&b)[1], ((__half2*)&c)[1]);
    ((__half2*)&c)[2] = __hfma2(((__half2*)&a)[2], ((__half2*)&b)[2], ((__half2*)&c)[2]);
    ((__half2*)&c)[3] = __hfma2(((__half2*)&a)[3], ((__half2*)&b)[3], ((__half2*)&c)[3]);
#else
#endif
}
template <>
__device__ __inline__ void fma_v4<float>(const int4 a, const int4 b, int4 &c){
    ((float*)&c)[0] = ((float*)&a)[0] * ((float*)&b)[0] + ((float*)&c)[0];
    ((float*)&c)[1] = ((float*)&a)[1] * ((float*)&b)[1] + ((float*)&c)[1];
    ((float*)&c)[2] = ((float*)&a)[2] * ((float*)&b)[2] + ((float*)&c)[2];
    ((float*)&c)[3] = ((float*)&a)[3] * ((float*)&b)[3] + ((float*)&c)[3];
}

template <typename T>
__device__ __inline__ int4 add_v4(const int4 a, const int4 b);

template <>
__device__ __inline__ int4 add_v4<__half>(const int4 a, const int4 b){
    int4 res = {0,0,0,0};
#if __CUDA_ARCH__ >= 600
    ((__half2*)&res)[0] = __hadd2(((__half2*)&a)[0], ((__half2*)&b)[0]);
    ((__half2*)&res)[1] = __hadd2(((__half2*)&a)[1], ((__half2*)&b)[1]);
    ((__half2*)&res)[2] = __hadd2(((__half2*)&a)[2], ((__half2*)&b)[2]);
    ((__half2*)&res)[3] = __hadd2(((__half2*)&a)[3], ((__half2*)&b)[3]);
#else
#endif
    return res;
}
template <>
__device__ __inline__ int4 add_v4<float>(const int4 a, const int4 b){
    int4 res = {0,0,0,0};
    ((float*)&res)[0] = ((float*)&a)[0] + ((float*)&b)[0];
    ((float*)&res)[1] = ((float*)&a)[1] + ((float*)&b)[1];
    ((float*)&res)[2] = ((float*)&a)[2] + ((float*)&b)[2];
    ((float*)&res)[3] = ((float*)&a)[3] + ((float*)&b)[3];
    return res;
}

template <typename T>
__inline__ __device__ T reduce_v4(int4 data){
    T res = (T)0;
    for(int i = 0; i < sizeof(int4)/sizeof(T); i++){
	res = Math<T,T,T>::add(res, ((T*)&data)[i]);
    }
}

template <typename T>
__device__ __inline__ void activation(const int activation, int4 &v){
    T *t_v = (T*)&v;
    constexpr int T_NUMS_PER_INT4 = sizeof(int4) / sizeof(T);
    if(activation ==1){
        for(int i = 0; i < T_NUMS_PER_INT4; i++)
            t_v[i] = Math<T,T,T>::ge(t_v[i], (T)0)?
		     t_v[i] : (T)0;
    } else{
        for(int i = 0; i < T_NUMS_PER_INT4; i++){
            T tmp = expf(t_v[i]);
            t_v[i]  = tmp * __frcp_rn(tmp + (T)1);
        }
    }
}
template <>
__device__ __inline__ void activation<__half>(const int activation, int4 &v){
#if __CUDA_ARCH__ >= 600
    __half2 *h2_v = (__half2*)&v;
    int    *int_v = (int*)&v;
    if(activation ==1){
        for(int i = 0; i < 4; i++)
            int_v[i] = __vmaxs2(int_v[i], 0);
    } else{
        __half2 one = {(__half)1.f, (__half)1.f};
        for(int i = 0; i < 4; i++){
            __half2 tmp = h2exp(h2_v[i]);
            h2_v[i] = __hmul2(tmp, h2rcp(__hadd2(one, tmp)));// __h2div(tmp, __hadd2(one, tmp));
        }
    }
#else
#endif
}
template<typename T>
__device__ __inline__ void clip(int4 &v, float clip_min, float clip_max){
    T *t_v = (T*)&v;
    constexpr int T_NUMS_PER_INT4 = sizeof(int4) / sizeof(T);
    for(int i = 0; i < T_NUMS_PER_INT4; i++){
        t_v[i] = Math<T,T,T>::ge(t_v[i], (T)clip_min)?
	         t_v[i] : (T)clip_min;
        t_v[i] = Math<T,T,T>::le(t_v[i], (T)clip_max)?
	         t_v[i] : (T)clip_max;
    }
}

//matrix: NxK
// N: pad int4
// K: pad int4
// layout and fuse pattern  consistent with gemm
//BLK_TILE_N: min:8
template<typename T, int BLK_TILE_N, int THD_TILE_N_V4, int BLK_SIZE>
__global__ void gemv(void *output, 
    const void *vec,
    const void *matrix,
    const void *bias,
    const int padK,
    const int padN,
    const fuse_param_t fuse_param)
{
    // blk conofig
    // one int4 per thd along K
    constexpr int T_NUMS_PER_INT4 = sizeof(int4) / sizeof(T);
    constexpr int BLK_TILE_N_V4 = BLK_TILE_N / T_NUMS_PER_INT4;
    constexpr int THD_TILE_N = THD_TILE_N_V4 * T_NUMS_PER_INT4;
    constexpr int BLK_SIZE_Y = BLK_TILE_N_V4 / THD_TILE_N_V4;
    constexpr int BLK_SIZE_X = BLK_SIZE / BLK_SIZE_Y;
    constexpr int BLK_TILE_K = BLK_SIZE_X;
    int pad_k_v4 = padK / T_NUMS_PER_INT4;
    int pad_n_v4 = padN / T_NUMS_PER_INT4;
    int n_id = blockIdx.x*BLK_TILE_N + threadIdx.y*T_NUMS_PER_INT4;

    int64_t b_base_v4 = (int64_t)n_id*pad_k_v4;
    int4 *matrix_base_v4 = (int4*)matrix + b_base_v4;
    int4 reg_c[THD_TILE_N];
    int4 reg_b[THD_TILE_N];
    bool in_n_range[THD_TILE_N_V4];
    int4 reg_a;
    int4 zero = {0,0,0,0};
    T c[THD_TILE_N] = { T(0) };
#pragma unroll
    for(int i = 0; i < THD_TILE_N; i++)    c[i] = (T)0;
#pragma unroll
    for(int i = 0; i < THD_TILE_N; i++){
	    reg_c[i] = zero;
    }
#pragma unroll
    for(int i = 0; i < THD_TILE_N_V4; i++){
	    in_n_range[i] = blockIdx.x*BLK_TILE_N_V4 + threadIdx.y + i*BLK_SIZE_Y < pad_n_v4;
    }


    // ld global VxM
#pragma unroll
    for(int k = 0; k < DivUp(pad_k_v4,BLK_TILE_K); k++){

	int64_t off = k*BLK_TILE_K + threadIdx.x;
	bool in_range = off < pad_k_v4;
        reg_a = in_range? ((int4*)vec)[off] : zero;
#pragma unroll
	for(int i = 0; i < THD_TILE_N_V4; i++){
#pragma unroll
	    for(int j = 0; j < T_NUMS_PER_INT4; j++){
	        reg_b[i*T_NUMS_PER_INT4 + j] = in_n_range[i] && in_range ? 
		                matrix_base_v4[(i*T_NUMS_PER_INT4*BLK_SIZE_Y+j)*pad_k_v4 + off]
				: zero;
	        fma_v4<T>(reg_a, reg_b[i*T_NUMS_PER_INT4 + j], 
			         reg_c[i*T_NUMS_PER_INT4 + j]);
	    }
	}
    }
    // int4 reduce to half
#pragma unroll
    for(int i = 0; i < THD_TILE_N; i++){
#pragma unroll
        for(int n = 0; n < T_NUMS_PER_INT4; n++){
            c[i] = Math<T,T,T>::add( ((T*)reg_c)[i*T_NUMS_PER_INT4 + n], 
			             c[i]);
        }
    }
    __shared__ T smem[BLK_SIZE_X*BLK_TILE_N];
    
    int reduce_off = (threadIdx.y*THD_TILE_N)*BLK_SIZE_X + threadIdx.x;
    constexpr int REDUCE_SIZE = BLK_SIZE_X;
    if(REDUCE_SIZE >= 64){
#pragma unroll
        for(int i = 0; i < THD_TILE_N; i++){
            smem[reduce_off + i*BLK_SIZE_X] = c[i];
        }
        __syncthreads();
    }

    //reduce
    if(REDUCE_SIZE >= 1024){
        if(threadIdx.x < 512)
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
	        smem[reduce_off + i*BLK_SIZE_X] = Math<T,T,T>::add(smem[reduce_off + i*BLK_SIZE_X], 
			                                     smem[512 + reduce_off + i*BLK_SIZE_X]);
        __syncthreads();
    }
    if(REDUCE_SIZE >= 512){
        if(threadIdx.x < 256)
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
	        smem[reduce_off + i*BLK_SIZE_X] = Math<T,T,T>::add(smem[reduce_off + i*BLK_SIZE_X], 
			                                     smem[256 + reduce_off + i*BLK_SIZE_X]);
        __syncthreads();
    }
    if(REDUCE_SIZE >= 256){
        if(threadIdx.x < 128)
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
	        smem[reduce_off + i*BLK_SIZE_X] = Math<T,T,T>::add(smem[reduce_off + i*BLK_SIZE_X], 
			                                     smem[128 + reduce_off + i*BLK_SIZE_X]);
        __syncthreads();
    }
    if(REDUCE_SIZE >= 128){
        if(threadIdx.x < 64)
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
	        smem[reduce_off + i*BLK_SIZE_X] = Math<T,T,T>::add(smem[reduce_off + i*BLK_SIZE_X], 
			                                      smem[64 + reduce_off + i*BLK_SIZE_X]);
        __syncthreads();
    }

    unsigned FULL_MASK = __activemask();
    if (REDUCE_SIZE >= 64) {
	if(threadIdx.x < 32){
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
                c[i] = Math<T,T,T>::add(smem[reduce_off + i*BLK_SIZE_X], 
            		            smem[reduce_off + i*BLK_SIZE_X + 32]);
        }
    }
    if(threadIdx.x < 32){
        if (REDUCE_SIZE >= 32) {
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
                c[i] = Math<T,T,T>::add(c[i], __shfl_down_sync(FULL_MASK, c[i], 16));
        }
        if (REDUCE_SIZE >= 16) {
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
                c[i] = Math<T,T,T>::add(c[i], __shfl_down_sync(FULL_MASK, c[i], 8));
        }
        if (REDUCE_SIZE >= 8) {
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
                c[i] = Math<T,T,T>::add(c[i], __shfl_down_sync(FULL_MASK, c[i], 4));
        }
        if (REDUCE_SIZE >= 4) {
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
                c[i] = Math<T,T,T>::add(c[i], __shfl_down_sync(FULL_MASK, c[i], 2));
        }
        if (REDUCE_SIZE >= 2) {
#pragma unroll
            for(int i = 0; i < THD_TILE_N; i++)
                c[i] = Math<T,T,T>::add(c[i], __shfl_down_sync(FULL_MASK, c[i], 1));
        }
    }

    // shared shuffle
    int4 *smem_v4 = (int4*)smem;
    if (threadIdx.x == 0) {
#pragma unroll
        for(int i = 0; i < THD_TILE_N_V4; i++){
            smem_v4[i*BLK_SIZE_Y + threadIdx.y] = ((int4*)c)[i];
	}
    }
    __syncthreads();

    int tid = threadIdx.y*BLK_SIZE_X + threadIdx.x;
    for(int thd_off = tid; thd_off < BLK_TILE_N_V4; thd_off += BLK_SIZE){
	int out_off = blockIdx.x*BLK_TILE_N_V4 + thd_off;
	bool in_output_range = out_off < pad_n_v4;

	if(in_output_range){
	    int4 bias_data = bias!=NULL? ((int4*)bias)[out_off] : zero;
	    //TODO add bias
	    int4 out = add_v4<T>(smem_v4[thd_off], bias_data);

	    // fuse
	    if(fuse_param.has_activation)    activation<T>(fuse_param.has_activation, out);
	    if(fuse_param.has_clip)          clip<T>(out, fuse_param.clip_min, fuse_param.clip_max);
            int concatV4_off = 0;
            if(fuse_param.has_concat){
                    int concat_offset_v4 = fuse_param.concat_offset / T_NUMS_PER_INT4;
                    int concat_stride_v4 = fuse_param.concat_stride / T_NUMS_PER_INT4;
                    concatV4_off = concat_offset_v4 + blockIdx.y*concat_stride_v4;
                    out_off += concatV4_off;
            }

	    ((int4*)output)[out_off] = out;
	}
    }
    
}


template<typename T>
ppl::common::RetCode PPLCUDAGemvForwardImp(
    const cudaStream_t &stream,
    const int M,
    const int padN,
    const int padK,
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ppl::nn::common::GemmParam &param,
    void* temp_buffer, 
    const fuse_param_t &fuse_param) 
{
    if(!param.transB)    return ppl::common::RC_UNSUPPORTED;

    constexpr int ELEM_NUM_PR_LD = sizeof(int4)/sizeof(T);
    constexpr int expect_blocks = 64;
    //constexpr int MAX_BLK_SIZE = 256;
    //constexpr int MAX_THD_TILE_N_V4 = 4;
    int n_v4 = padN / ELEM_NUM_PR_LD;
    int blk_tile_n_v4 = DivUp(n_v4, expect_blocks/M);
#define LAUNCH_KERNEL(){ \
    constexpr int BLK_TILE_N = BLK_SIZE_Y * THD_TILE_N_V4 * ELEM_NUM_PR_LD; \
    constexpr int BLK_SIZE = BLK_SIZE_Y * BLK_SIZE_X; \
    dim3 grid; \
    grid.x = DivUp(padN, BLK_TILE_N); \
    grid.y = 1;    grid.z = 1; \
    dim3 threads = dim3(BLK_SIZE_X, BLK_SIZE_Y,1); \
    gemv<T, BLK_TILE_N, THD_TILE_N_V4, BLK_SIZE><<<grid, threads, 0, stream>>>\
    	(output, input, weight, bias, padK, padN, fuse_param); \
}
#define CONFIG_KERNEL(_blk_tile_n_v4){ \
    if(BLK_SIZE_X <= 64 && blk_tile_n_v4 >= 16){ \
        constexpr int THD_TILE_N_V4 = 4; \
        constexpr int BLK_SIZE_Y = 4; \
        LAUNCH_KERNEL(); \
    } else if(blk_tile_n_v4 >= 8){ \
        constexpr int THD_TILE_N_V4 = 4; \
        constexpr int BLK_SIZE_Y = 2; \
        LAUNCH_KERNEL(); \
    } else if(blk_tile_n_v4 >= 4){ \
        constexpr int THD_TILE_N_V4 = 2; \
        constexpr int BLK_SIZE_Y = 2; \
        LAUNCH_KERNEL(); \
    } else if(blk_tile_n_v4 >= 2){ \
        constexpr int THD_TILE_N_V4 = 2; \
        constexpr int BLK_SIZE_Y = 1; \
        LAUNCH_KERNEL(); \
    } else{ \
        constexpr int THD_TILE_N_V4 = 1; \
        constexpr int BLK_SIZE_Y = 1; \
        LAUNCH_KERNEL(); \
    } \
}
    if (padK >= 512){
        constexpr int BLK_SIZE_X = 64;
        CONFIG_KERNEL(blk_tile_n_v4);
    }
    else{
	constexpr int BLK_SIZE_X = 32;
	CONFIG_KERNEL(blk_tile_n_v4);
    }

    return ppl::common::RC_SUCCESS;
}
