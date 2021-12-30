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

#ifndef PPLCUDA_MATH_MATH_H_
#define PPLCUDA_MATH_MATH_H_

#include <cuda_fp16.h>
#include <cuda.h>

struct PPLHalf4 {
    half2 x;
    half2 y;
};

typedef PPLHalf4 half4;

struct PPLHalf8 {
    half4 x;
    half4 y;
};
typedef PPLHalf8 half8;

struct PPLFloat8 {
    float4 x;
    float4 y;
};
#if (__CUDA_ARCH__ >= 530)
#define PPL_DEVICE_FP16
#endif

typedef PPLFloat8 float8;

template <typename T>
__host__ __device__ const T &CudaMin(const T &a, const T &b)
{
    return b < a ? b : a;
}

template <typename T, typename RT, typename AT>
struct Math {
    static inline __device__ RT add(T lhs, T rhs)
    {
        return lhs + rhs;
    }
    static inline __device__ RT sub(T lhs, T rhs)
    {
        return lhs - rhs;
    }
    static inline __device__ RT mul(T lhs, T rhs)
    {
        return lhs * rhs;
    }
    static inline __device__ RT div(T lhs, T rhs)
    {
        return lhs / rhs;
    }
    static inline __device__ RT neg(T v)
    {
        return -v;
    }
    static inline __device__ bool lt(T lhs, T rhs)
    {
        return lhs < rhs;
    }
    static inline __device__ bool le(T lhs, T rhs)
    {
        return lhs <= rhs;
    }
    static inline __device__ bool gt(T lhs, T rhs)
    {
        return lhs > rhs;
    }
    static inline __device__ bool ge(T lhs, T rhs)
    {
        return lhs >= rhs;
    }
    static inline __device__ bool eq(T lhs, T rhs)
    {
        return lhs == rhs;
    }
    static inline __device__ RT zero()
    {
        return (RT)0;
    }
};

template <>
struct Math<half, half, half> {
    static inline __device__ half add(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hadd(lhs, rhs);
#else
        return __float2half(__half2float(lhs) + __half2float(rhs));
#endif
    }
    static inline __device__ half sub(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hsub(lhs, rhs);
#else
        return __float2half(__half2float(lhs) - __half2float(rhs));
#endif
    }
    static inline __device__ half mul(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hmul(lhs, rhs);
#else
        return __float2half(__half2float(lhs) * __half2float(rhs));
#endif
    }
    static inline __device__ half div(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hdiv(lhs, rhs);
#else
        return __float2half(__half2float(lhs) / __half2float(rhs));
#endif
    }
    static inline __device__ half neg(half v)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hneg(v);
#else
        return __float2half(-(__half2float(v)));
#endif
    }
    static inline __device__ bool lt(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hlt(lhs, rhs);
#else
        return __half2float(lhs) < __half2float(rhs);
#endif
    }

    static inline __device__ bool le(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hle(lhs, rhs);
#else
        return __half2float(lhs) <= __half2float(rhs);
#endif
    }
    static inline __device__ bool gt(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hgt(lhs, rhs);
#else
        return __half2float(lhs) > __half2float(rhs);
#endif
    }
    static inline __device__ bool ge(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hge(lhs, rhs);
#else
        return __half2float(lhs) >= __half2float(rhs);
#endif
    }
    static inline __device__ bool eq(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __heq(lhs, rhs);
#else
        return __half2float(lhs) == __half2float(rhs);
#endif
    }
    static inline __device__ half zero()
    {
#if (__CUDA_ARCH__ >= 530)
        return (half)0;
#else
        return __float2half(0.0);
#endif
    }
};

template <>
struct Math<half, half, float> {
    static inline __device__ half add(half lhs, half rhs)
    {
        return __float2half(__half2float(lhs) + __half2float(rhs));
    }
    static inline __device__ half sub(half lhs, half rhs)
    {
        return __float2half(__half2float(lhs) - __half2float(rhs));
    }
    static inline __device__ half mul(half lhs, half rhs)
    {
        return __float2half(__half2float(lhs) * __half2float(rhs));
    }
    static inline __device__ half div(half lhs, half rhs)
    {
        return __float2half(__half2float(lhs) / __half2float(rhs));
    }

    static inline __device__ float add(float lhs, half rhs)
    {
        return (lhs + __half2float(rhs));
    }
    static inline __device__ float sub(float lhs, half rhs)
    {
        return (lhs - __half2float(rhs));
    }
    static inline __device__ float mul(float lhs, half rhs)
    {
        return (lhs * __half2float(rhs));
    }
    static inline __device__ float div(float lhs, half rhs)
    {
        return (lhs / __half2float(rhs));
    }

    static inline __device__ float add(float lhs, float rhs)
    {
        return (lhs + (rhs));
    }
    static inline __device__ float sub(float lhs, float rhs)
    {
        return (lhs - (rhs));
    }
    static inline __device__ float mul(float lhs, float rhs)
    {
        return (lhs * (rhs));
    }
    static inline __device__ float div(float lhs, float rhs)
    {
        return (lhs / (rhs));
    }

    static inline __device__ half neg(half v)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hneg(v);
#else
        return __float2half(-(__half2float(v)));
#endif
    }
    static inline __device__ float neg(float v)
    {
        return -v;
    }

    static inline __device__ bool lt(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hlt(lhs, rhs);
#else
        return __half2float(lhs) < __half2float(rhs);
#endif
    }
    static inline __device__ bool le(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hle(lhs, rhs);
#else
        return __half2float(lhs) <= __half2float(rhs);
#endif
    }
    static inline __device__ bool gt(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hgt(lhs, rhs);
#else
        return __half2float(lhs) > __half2float(rhs);
#endif
    }
    static inline __device__ bool ge(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hge(lhs, rhs);
#else
        return __half2float(lhs) >= __half2float(rhs);
#endif
    }
    static inline __device__ bool eq(half lhs, half rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __heq(lhs, rhs);
#else
        return __half2float(lhs) == __half2float(rhs);
#endif
    }
    static inline __device__ half zero()
    {
#if (__CUDA_ARCH__ >= 530)
        return (half)0;
#else
        return __float2half(0.0);
#endif
    }
};

template <>
struct Math<half2, half2, half2> {
    static inline __device__ half2 add(half2 lhs, half2 rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hadd2(lhs, rhs);
#else
        float2 lhs_f = __half22float2(lhs);
        float2 rhs_f = __half22float2(rhs);

        lhs_f.x = lhs_f.x + rhs_f.x;
        lhs_f.y = lhs_f.y + rhs_f.y;

        return __float22half2_rn(lhs_f);
#endif
    }
    static inline __device__ half2 sub(half2 lhs, half2 rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hsub2(lhs, rhs);
#else
        float2 lhs_f = __half22float2(lhs);
        float2 rhs_f = __half22float2(rhs);

        lhs_f.x = lhs_f.x - rhs_f.x;
        lhs_f.y = lhs_f.y - rhs_f.y;

        return __float22half2_rn(lhs_f);
#endif
    }
    static inline __device__ half2 mul(half2 lhs, half2 rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hmul2(lhs, rhs);
#else
        float2 lhs_f = __half22float2(lhs);
        float2 rhs_f = __half22float2(rhs);

        lhs_f.x = lhs_f.x * rhs_f.x;
        lhs_f.y = lhs_f.y * rhs_f.y;

        return __float22half2_rn(lhs_f);
#endif
    }
    static inline __device__ half2 div(half2 lhs, half2 rhs)
    {
#if (__CUDA_ARCH__ >= 530)
        return __h2div(lhs, rhs);
#else
        float2 lhs_f = __half22float2(lhs);
        float2 rhs_f = __half22float2(rhs);

        lhs_f.x = lhs_f.x / rhs_f.x;
        lhs_f.y = lhs_f.y / rhs_f.y;

        return __float22half2_rn(lhs_f);
#endif
    }
    static inline __device__ half2 neg(half2 v)
    {
#if (__CUDA_ARCH__ >= 530)
        return __hneg2(v);
#else
        float2 vf = __half22float2(v);
        vf.x      = -vf.x;
        vf.y      = -vf.y;

        return __float22half2_rn(vf);
#endif
    }
    static inline __device__ half2 zero()
    {
        return __half2half2(Math<half, half, half>::zero());
    }
};

template <>
struct Math<half4, half4, half4> {
    static inline __device__ half4 add(half4 lhs, half4 rhs)
    {
        half4 res;
        res.x = Math<half2, half2, half2>::add(lhs.x, rhs.x);
        res.y = Math<half2, half2, half2>::add(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half4 sub(half4 lhs, half4 rhs)
    {
        half4 res;
        res.x = Math<half2, half2, half2>::sub(lhs.x, rhs.x);
        res.y = Math<half2, half2, half2>::sub(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half4 mul(half4 lhs, half4 rhs)
    {
        half4 res;
        res.x = Math<half2, half2, half2>::mul(lhs.x, rhs.x);
        res.y = Math<half2, half2, half2>::mul(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half4 div(half4 lhs, half4 rhs)
    {
        half4 res;
        res.x = Math<half2, half2, half2>::div(lhs.x, rhs.x);
        res.y = Math<half2, half2, half2>::div(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half4 neg(half4 v)
    {
        half4 res;
        res.x = Math<half2, half2, half2>::neg(v.x);
        res.y = Math<half2, half2, half2>::neg(v.y);
        return res;
    }
    static inline __device__ half4 zero()
    {
        half4 res;
        res.x = Math<half2, half2, half2>::zero();
        res.y = Math<half2, half2, half2>::zero();
        return res;
    }
};

template <>
struct Math<half8, half8, half8> {
    static inline __device__ half8 add(half8 lhs, half8 rhs)
    {
        half8 res;
        res.x = Math<half4, half4, half4>::add(lhs.x, rhs.x);
        res.y = Math<half4, half4, half4>::add(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half8 sub(half8 lhs, half8 rhs)
    {
        half8 res;
        res.x = Math<half4, half4, half4>::sub(lhs.x, rhs.x);
        res.y = Math<half4, half4, half4>::sub(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half8 mul(half8 lhs, half8 rhs)
    {
        half8 res;
        res.x = Math<half4, half4, half4>::mul(lhs.x, rhs.x);
        res.y = Math<half4, half4, half4>::mul(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half8 div(half8 lhs, half8 rhs)
    {
        half8 res;
        res.x = Math<half4, half4, half4>::div(lhs.x, rhs.x);
        res.y = Math<half4, half4, half4>::div(lhs.y, rhs.y);
        return res;
    }
    static inline __device__ half8 neg(half8 v)
    {
        half8 res;
        res.x = Math<half4, half4, half4>::neg(v.x);
        res.y = Math<half4, half4, half4>::neg(v.y);
        return res;
    }
    static inline __device__ half8 zero()
    {
        half8 res;
        res.x = Math<half4, half4, half4>::zero();
        res.y = Math<half4, half4, half4>::zero();
        return res;
    }
};
#endif