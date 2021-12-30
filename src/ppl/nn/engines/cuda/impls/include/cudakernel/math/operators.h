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

#ifndef PPLCUDA_MATH_OP_H_
#define PPLCUDA_MATH_OP_H_

#include "cudakernel/math/math.h"

#define HALF_MAX 65504
#define HLAF_MIN -65504
#define FLT_MAX  3.40282346638528859811704183484516925e+38F
#define FLT_MIN  -3.40282346638528859811704183484516925e+38F

template <typename T>
T getMin();
template <>
inline __host__ __device__ int64_t getMin<int64_t>()
{
    return INT64_MIN;
}
template <>
inline __host__ __device__ float getMin<float>()
{
    return FLT_MIN;
}
template <>
inline __host__ __device__ half getMin<half>()
{
    return HLAF_MIN;
}
template <>
inline __host__ __device__ int8_t getMin<int8_t>()
{
    return INT8_MIN;
}

template <typename T>
T getMax();
template <>
inline __host__ __device__ int64_t getMax<int64_t>()
{
    return INT64_MAX;
}
template <>
inline __host__ __device__ float getMax<float>()
{
    return FLT_MAX;
}
template <>
inline __host__ __device__ half getMax<half>()
{
    return HALF_MAX;
}
template <>
inline __host__ __device__ int8_t getMax<int8_t>()
{
    return INT8_MAX;
}

template <typename T>
T getZero();
template <>
inline __host__ __device__ float getZero()
{
    return (float)0.0;
}
template <>
inline __host__ __device__ half getZero()
{
    return (half)0.0;
}
template <>
inline __host__ __device__ int64_t getZero()
{
    return (int64_t)0;
}
template <>
inline __host__ __device__ int8_t getZero()
{
    return (int8_t)0;
}

template <typename T>
T getOne();
template <>
inline __host__ __device__ float getOne()
{
    return (float)1.0;
}
template <>
inline __host__ __device__ half getOne()
{
    return (half)1.0;
}
template <>
inline __host__ __device__ int64_t getOne()
{
    return (int64_t)1;
}
template <>
inline __host__ __device__ int8_t getOne()
{
    return (int8_t)1;
}
template <typename src_type, typename dst_type, typename acc_type>
struct SumOp {
    typedef acc_type acctype;
    typedef src_type srctype;
    typedef dst_type dsttype;
    typedef Math<src_type, dst_type, acc_type> OpMath;
    src_type* src;
    dst_type* dst;
    static __device__ acc_type compute(acc_type lhs, acc_type rhs)
    {
        return Math<src_type, dst_type, acc_type>::add(lhs, rhs);
    }
    static __host__ __device__ acc_type InitVal()
    {
        return getZero<acc_type>();
    }
    __host__ __device__ acc_type fetch(int64_t idx)
    {
        return src[idx];
    }
    __host__ __device__ void out(int64_t idx, dst_type val)
    {
        dst[idx] = val;
    }
    __host__ __device__ SumOp(src_type* src, dst_type* dst)
        : src(src)
        , dst(dst) {}

    const static int type = 0;
};

template <typename src_type, typename dst_type, typename acc_type>
struct MaxOp {
    typedef acc_type acctype;
    typedef src_type srctype;
    typedef dst_type dsttype;
    typedef Math<src_type, dst_type, acc_type> OpMath;
    src_type* src;
    dst_type* dst;
    static __device__ acc_type compute(acc_type lhs, acc_type rhs)
    {
        return Math<src_type, dst_type, acc_type>::gt(lhs, rhs) ? lhs : rhs;
    }
    static __host__ __device__ acc_type InitVal()
    {
        return getMin<acc_type>();
    }
    __host__ __device__ acc_type fetch(int64_t idx)
    {
        return src[idx];
    }
    __host__ __device__ void out(int64_t idx, dst_type val)
    {
        dst[idx] = val;
    }
    __host__ __device__ MaxOp(src_type* src, dst_type* dst)
        : src(src)
        , dst(dst) {}
    const static int type = 1;
};

template <typename src_type, typename dst_type, typename acc_type>
struct MinOp {
    typedef acc_type acctype;
    typedef src_type srctype;
    typedef dst_type dsttype;
    typedef Math<src_type, dst_type, acc_type> OpMath;
    src_type* src;
    dst_type* dst;
    static __device__ acc_type compute(acc_type lhs, acc_type rhs)
    {
        return Math<src_type, dst_type, acc_type>::gt(lhs, rhs) ? rhs : lhs;
    }
    static __host__ __device__ acc_type InitVal()
    {
        return getMax<acc_type>();
    }
    __host__ __device__ acc_type fetch(int64_t idx)
    {
        return src[idx];
    }
    __host__ __device__ void out(int64_t idx, dst_type val)
    {
        dst[idx] = val;
    }
    __host__ __device__ MinOp(src_type* src, dst_type* dst)
        : src(src)
        , dst(dst) {}
    const static int type = 2;
};

template <typename src_type, typename dst_type, typename acc_type>
struct ProdOp {
    typedef acc_type acctype;
    typedef src_type srctype;
    typedef dst_type dsttype;
    typedef Math<src_type, dst_type, acc_type> OpMath;
    src_type* src;
    dst_type* dst;
    static __device__ acc_type compute(acc_type lhs, acc_type rhs)
    {
        return Math<src_type, dst_type, acc_type>::mul(lhs, rhs);
    }
    static __host__ __device__ acc_type InitVal()
    {
        return getOne<acc_type>();
    }
    __host__ __device__ acc_type fetch(int64_t idx)
    {
        return src[idx];
    }
    __host__ __device__ void out(int64_t idx, dst_type val)
    {
        dst[idx] = val;
    }
    __host__ __device__ ProdOp(src_type* src, dst_type* dst)
        : src(src)
        , dst(dst) {}
    const static int type = 4;
};

#endif