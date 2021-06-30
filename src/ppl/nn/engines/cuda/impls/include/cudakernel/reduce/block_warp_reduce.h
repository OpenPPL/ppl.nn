#ifndef PPLCUDA_REDUCE_BLOCK_WARP_REDUCE_H_
#define PPLCUDA_REDUCE_BLOCK_WARP_REDUCE_H_
#include "cudakernel/common/atomic.h"
#include "cudakernel/math/operators.h"

template <typename T, class Operator, int ReduceSize>
__inline__ __device__ void warp_reduce_unroll(Operator op, T* shared_data, T& val)
{
    int tid                      = threadIdx.x;
    constexpr unsigned FULL_MASK = 0xffffffff;
    if (ReduceSize >= 64 && (shared_data != nullptr)) {
        shared_data[tid] = op.compute(shared_data[tid], shared_data[tid + 32]);
    }
    val = (shared_data == nullptr) ? val : shared_data[tid];
    if (ReduceSize >= 32) {
        val = op.compute(val, __shfl_down_sync(FULL_MASK, val, 16));
    }
    if (ReduceSize >= 16) {
        val = op.compute(val, __shfl_down_sync(FULL_MASK, val, 8));
    }
    if (ReduceSize >= 8) {
        val = op.compute(val, __shfl_down_sync(FULL_MASK, val, 4));
    }
    if (ReduceSize >= 4) {
        val = op.compute(val, __shfl_down_sync(FULL_MASK, val, 2));
    }
    if (ReduceSize >= 2) {
        val = op.compute(val, __shfl_down_sync(FULL_MASK, val, 1));
    }
    if (shared_data != nullptr) {
        shared_data[tid] = val;
    }
}

template <typename T, class Operator, int ReduceSize>
__inline__ __device__ void block_reduce_row(Operator op, T* shared_data, T& val)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (ReduceSize >= 1024) {
        if (tid < 512)
            shared_data[tid] = op.compute(shared_data[tid], shared_data[tid + 512]);
        __syncthreads();
    }
    if (ReduceSize >= 512) {
        if (tid < 256)
            shared_data[tid] = op.compute(shared_data[tid], shared_data[tid + 256]);
        __syncthreads();
    }
    if (ReduceSize >= 256) {
        if (tid < 128)
            shared_data[tid] = op.compute(shared_data[tid], shared_data[tid + 128]);
        __syncthreads();
    }
    if (ReduceSize >= 128) {
        if (tid < 64)
            shared_data[tid] = op.compute(shared_data[tid], shared_data[tid + 64]);
        __syncthreads();
    }
    if (tid < 32) {
        warp_reduce_unroll<T, Operator, ReduceSize>(op, shared_data, val);
    }
    __syncthreads();
}

#endif