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

#ifndef PPLCUDA_COMMON_ATOMIC_H_
#define PPLCUDA_COMMON_ATOMIC_H_

#include <cuda_fp16.h>
#include <cuda.h>
#include "cuda_arch.h"

//
// ref: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh
//

template <typename T>
__device__ void atomic_max(T *addr, T val)
{
    atomicMax(addr, val);
}

template <typename T>
__device__ void atomic_add(T *addr, T val)
{
    atomicAdd(addr, val);
}
template <typename T>
__device__ void atomic_min(T *addr, T val)
{
    atomicMin(addr, val);
}

__device__ __inline__ void atomic_max(int8_t *addr, int8_t val)
{
    if (*addr >= val)
        return;

    unsigned int *const addr_as_ull = (unsigned int *)addr;
    unsigned int old               = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (reinterpret_cast<int8_t&>(assumed) >= val)
            break;
        old = atomicCAS(addr_as_ull, assumed, val);
    } while (assumed != old);
}

__device__ __inline__ void atomic_add(int8_t *addr, int8_t val)
{
    if (*addr >= val)
        return;

    unsigned int *const addr_as_ull = (unsigned int *)addr;
    unsigned int old               = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, reinterpret_cast<int8_t&>(old) + val);
    } while (assumed != old);
}

__device__ __inline__ void atomic_min(int8_t *addr, int8_t val)
{
    if (*addr >= val)
        return;

    unsigned int *const addr_as_ull = (unsigned int *)addr;
    unsigned int old               = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (reinterpret_cast<int8_t&>(assumed) <= val)
            break;
        old = atomicCAS(addr_as_ull, assumed, val);
    } while (assumed != old);
}

__device__ __inline__ void atomic_max(int64_t *addr, int64_t val)
{
    if (*addr >= val)
        return;

    unsigned long long *const addr_as_ull = (unsigned long long *)addr;
    unsigned long long old                = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (reinterpret_cast<int64_t &>(assumed) >= val)
            break;
        old = atomicCAS(addr_as_ull, assumed, val);
    } while (assumed != old);
}

__device__ __inline__ void atomic_add(int64_t *addr, int64_t val)
{
    if (*addr >= val)
        return;

    unsigned long long *const addr_as_ull = (unsigned long long *)addr;
    unsigned long long old                = *addr_as_ull, assumed;
    do {
        assumed = old;
        old     = atomicCAS(addr_as_ull, assumed, reinterpret_cast<int64_t &>(old) + val);
    } while (assumed != old);
}

__device__ __inline__ void atomic_min(int64_t *addr, int64_t val)
{
    if (*addr >= val)
        return;

    unsigned long long *const addr_as_ull = (unsigned long long *)addr;
    unsigned long long old                = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (reinterpret_cast<int64_t &>(assumed) <= val)
            break;
        old = atomicCAS(addr_as_ull, assumed, val);
    } while (assumed != old);
}

__device__ __inline__ void atomic_max(float *addr, float val)
{
    if (*addr >= val)
        return;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old               = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val)
            break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

__device__ __inline__ void atomic_min(float *addr, float val)
{
    if (*addr <= val)
        return;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old               = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) <= val)
            break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

__device__ __inline__ void atomic_add(half *address, half val)
{
#if (__CUDA_ARCH__ >= 700) && (PPL_CUDA_NVCC_VERSION >= 100010)
    atomicAdd(address, val);
#else
    unsigned int *base_address = (unsigned int *)((char *)address - ((size_t)address & 2));
    unsigned int old           = *base_address;
    unsigned int assumed;
    unsigned short x;

    do {
        assumed = old;
        x       = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        x       = __half_as_short(__float2half(__half2float(*reinterpret_cast<const __half *>(&x)) + __half2float(val)));
        old     = (size_t)address & 2 ? (old & 0xffff) | (x << 16) : (old & 0xffff0000) | x;
        old     = atomicCAS(base_address, assumed, old);
    } while (assumed != old);
#endif
}

__device__ __inline__ void atomic_max(half *address, half val)
{
#if (__CUDA_ARCH__ >= 700) && (PPL_CUDA_NVCC_VERSION >= 100010)
    if (__hge(*address, val))
        return;
    unsigned short int *const addr_as_usi = (unsigned short int *)address;
    unsigned short int old                = *addr_as_usi, assumed;
    do {
        assumed = old;
        if (__hge(__ushort_as_half(assumed), val))
            break;
        old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val));
    } while (assumed != old);
#endif
}

__device__ __inline__ void atomic_min(half *address, half val)
{
#if (__CUDA_ARCH__ >= 700) && (PPL_CUDA_NVCC_VERSION >= 100010)
    if (__hle(*address, val))
        return;

    unsigned short int *const addr_as_usi = (unsigned short int *)address;
    unsigned short int old                = *addr_as_usi, assumed;
    do {
        assumed = old;
        if (__hle(__ushort_as_half(assumed), val))
            break;
        old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val));
    } while (assumed != old);
#endif
}

// Only Support Add/Max/Min
template <typename T, int Type>
struct AtomicOp {
    __device__ __inline__ void Compute(T *address, T val) {}
};

template <typename T>
struct AtomicOp<T, 0> {
    __device__ __inline__ void Compute(T *address, T val)
    {
        atomic_add(address, val);
    }
};

template <typename T>
struct AtomicOp<T, 1> {
    __device__ __inline__ void Compute(T *address, T val)
    {
        atomic_max(address, val);
    }
};

template <typename T>
struct AtomicOp<T, 2> {
    __device__ __inline__ void Compute(T *address, T val)
    {
        atomic_min(address, val);
    }
};

template <typename T, class Operator>
__device__ __inline__ void PPLAtomicWrite(T *output, T val, Operator op)
{
    AtomicOp<T, op.type> atomic_op;
    atomic_op.Compute(output, val);
}

#endif
