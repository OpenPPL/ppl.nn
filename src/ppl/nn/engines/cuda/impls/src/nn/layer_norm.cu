#include "cudakernel/common/common.cuh"
#include "cudakernel/nn/layer_norm.h"
#include <cuda_fp16.h>
#include <float.h>
/*
Par
in : [B, N]
alpha : [N]
beta : [N]
out : [B, N]

thread
grid : [B, 1, 1]
block : max(N, 1024)
*/
template<typename T, typename TPar>
__global__ void LayerNormKernel(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int B,
                                const int N, const T eps = 1e-5) {
    auto cur_in = in + blockIdx.x * N;
    auto cur_out = out + blockIdx.x * N;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    //BlockReduceSum
    sum = BlockReduceSum(sum);
    float mean = sum.x / N;
    float rstd = rsqrtf(sum.y / N - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        cur_out[tid] = T(((float)__ldg(cur_in + tid) - mean) * rstd * 
                            (float)__ldg(alpha + tid) + (float)__ldg(beta + tid));
    }
}

template<typename T, typename TPar>
__global__ void __launch_bounds__(32)
    LayerNormKernel32(const T* in, const TPar* alpha,
                      const TPar* beta, T* out, const int B,
                      const int N, const T eps = 1e-5) {
    auto cur_in = in + blockIdx.x * N;
    auto cur_out = out + blockIdx.x * N;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    //BlockReduceSum
    sum = BlockReduceSum(sum);
    float mean = sum.x / N;
    float rstd = rsqrtf(sum.y / N - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        cur_out[tid] = T(((float)__ldg(cur_in + tid) - mean) * rstd * 
                            (float)__ldg(alpha + tid) + (float)__ldg(beta + tid));
    }
}

template<typename T, typename TPar>
__global__ void __launch_bounds__(64)
    LayerNormKernel64(const T* in, const TPar* alpha,
                      const TPar* beta, T* out, const int B,
                      const int N, const T eps = 1e-5) {
    auto cur_in = in + blockIdx.x * N;
    auto cur_out = out + blockIdx.x * N;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    //BlockReduceSum
    sum = BlockReduceSum(sum);
    float mean = sum.x / N;
    float rstd = rsqrtf(sum.y / N - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        cur_out[tid] = T(((float)__ldg(cur_in + tid) - mean) * rstd * 
                            (float)__ldg(alpha + tid) + (float)__ldg(beta + tid));
    }
}

template<typename TPar>
ppl::common::RetCode PPLCUDALayernormForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    const TPar* alpha, 
    const TPar* beta,
    const int eps) {
    const int B = output_shape->GetDim(0);
    const int N = output_shape->GetDim(1);
    dim3 grid(B, 1, 1);
    if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        if(N < 512) {
            dim3 block(32, 1, 1);
            LayerNormKernel32<float, TPar><<<grid, block, 0, stream>>>
                ((const float*)input, alpha, beta, (float*)output, B, N, eps);
        } else if(N < 2048) {
            dim3 block(64, 1, 1);
            LayerNormKernel64<float, TPar><<<grid, block, 0, stream>>>
                ((const float*)input, alpha, beta, (float*)output, B, N, eps);
        } else {
            dim3 block(1024, 1, 1);
            LayerNormKernel<float, TPar><<<grid, block, 0, stream>>>
                ((const float*)input, alpha, beta, (float*)output, B, N, eps);
        }
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if(N < 512) {
            dim3 block(32, 1, 1);
            LayerNormKernel32<half, TPar><<<grid, block, 0, stream>>>
                ((const half*)input, alpha, beta, (half*)output, B, N, eps);
        } else if(N < 2048) {
            dim3 block(64, 1, 1);
            LayerNormKernel64<half, TPar><<<grid, block, 0, stream>>>
                ((const half*)input, alpha, beta, (half*)output, B, N, eps);
        } else {
            dim3 block(1024, 1, 1);
            LayerNormKernel<half, TPar><<<grid, block, 0, stream>>>
                ((const half*)input, alpha, beta, (half*)output, B, N, eps);
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;

}
