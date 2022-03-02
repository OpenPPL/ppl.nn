#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <memory>

template<typename T>
__device__ __forceinline__ T _Exp(T a);

template<>
__device__ __forceinline__ float _Exp<float>(float a) {
    return expf(a);
}

template<>
__device__ __forceinline__ double _Exp<double>(double a) {
    return exp(a);
}

template<>
__device__ __forceinline__ half _Exp<half>(half a) {
    return __float2half(exp(__half2float(a)));
}

template<typename T>
__device__ __forceinline__ T _Ldg(const T* p) {
    return __ldg(p);
}

template<>
__device__ __forceinline__ bool _Ldg<bool>(const bool* p) {
    return *p;
}

template<typename T>
__device__ __forceinline__ T _ExpMax() {
    return (T)20.0f;
}

template<>
__device__ __forceinline__ float _ExpMax<float>() {
    return 80.0f;
}

template<>
__device__ __forceinline__ double _ExpMax<double>() {
    return 800.0;
}

template<typename T>
__device__ __forceinline__ T CudaLogZero() {
    return (T)-_ExpMax<T>();
}

template<typename T>
__device__ __forceinline__ T _SafeExp(const T v) {
    return _Exp(min(v, _ExpMax<T>()));
}

template<typename T>
__device__ __forceinline__ T _LogAdd(const T x, const T y) {
    return x + max(log(_SafeExp(y - x) + (T)1.0f), y - x);
}

#define FINAL_MASK 0xffffffff
template<typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane,
                                        int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}


template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask,
                                            int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template<typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += WARP_SHFL_XOR(val, mask, 32, FINAL_MASK);
    return val;
}

template<typename T>
__device__ __forceinline__ T WarpReduceLogAddSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = _LogAdd(WARP_SHFL_XOR(val, mask, 32, FINAL_MASK), val);
    return val;
}


template<typename T>
__device__ __forceinline__ T BlockReduceSum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduceSum(val);
    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (lane < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = WarpReduceSum(val);
    return val;
}

inline int GetBlockSize(const int n, const int max_size = 1024) {
    int ret = 32;
    while(ret < n && ret < max_size) {
        ret <<= 1;
    }
    return ret;
}
