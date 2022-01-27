
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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_LEAKY_RELU_LEAKY_RELU_KERNEL_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_LEAKY_RELU_LEAKY_RELU_KERNEL_H_

#include <string>
#include <riscv-vector.h>

namespace ppl { namespace kernel { namespace riscv {

//
template <typename T, int32_t v_len>
struct register_v_helper;

template <typename T, int32_t v_len>
using register_v = typename register_v_helper<T, v_len>::U;

template <>
struct register_v_helper<float, 128> {
    typedef float32xm1_t U;
};
template <>
struct register_v_helper<__fp16, 128> {
    typedef float16xm1_t U;
};

//
template <typename T, int32_t v_len>
inline uint64_t vsetvli(const int32_t c_blk);

template <>
inline uint64_t vsetvli<float, 128>(const int32_t c_blk)
{
    return vsetvli(c_blk, RVV_E32, RVV_M1);
}
template <>
inline uint64_t vsetvli<__fp16, 128>(const int32_t c_blk)
{
    return vsetvli(c_blk, RVV_E16, RVV_M1);
}

//
template <typename T, int32_t v_len>
inline register_v<T, v_len> vlev(const T* addr, uint64_t n);

template <>
inline float32xm1_t vlev<float, 128>(const float* addr, uint64_t n)
{
    return vlev_float32xm1(addr, n);
}
template <>
inline float16xm1_t vlev<__fp16, 128>(const __fp16* addr, uint64_t n)
{
    return vlev_float16xm1(addr, n);
}

//
template <typename T, int32_t v_len>
inline void vsev(T* addr, register_v<T, v_len> va, uint64_t n);

template <>
inline void vsev<float, 128>(float* addr, float32xm1_t va, uint64_t n)
{
    return vsev_float32xm1(addr, va, n);
}
template <>
inline void vsev<__fp16, 128>(__fp16* addr, float16xm1_t va, uint64_t n)
{
    return vsev_float16xm1(addr, va, n);
}

//
template <typename T, int32_t v_len>
inline register_v<T, v_len> vmaxvf(register_v<T, v_len> va, T a, uint64_t n);

template <>
inline float32xm1_t vmaxvf<float, 128>(float32xm1_t va, float a, uint64_t n)
{
    return vfmaxvf_float32xm1(va, a, n);
}
template <>
inline float16xm1_t vmaxvf<__fp16, 128>(float16xm1_t va, __fp16 a, uint64_t n)
{
    return vfmaxvf_float16xm1(va, a, n);
}

//
template <typename T, int32_t v_len>
inline register_v<T, v_len> vminvf(register_v<T, v_len> va, T a, uint64_t n);

template <>
inline float32xm1_t vminvf<float, 128>(float32xm1_t va, float a, uint64_t n)
{
    return vfminvf_float32xm1(va, a, n);
}
template <>
inline float16xm1_t vminvf<__fp16, 128>(float16xm1_t va, __fp16 a, uint64_t n)
{
    return vfminvf_float16xm1(va, a, n);
}

//
template <typename T, int32_t v_len>
inline register_v<T, v_len> vmulvf(register_v<T, v_len> va, T a, uint64_t n);

template <>
inline float32xm1_t vmulvf<float, 128>(float32xm1_t va, float a, uint64_t n)
{
    return vfmulvf_float32xm1(va, a, n);
}
template <>
inline float16xm1_t vmulvf<__fp16, 128>(float16xm1_t va, __fp16 a, uint64_t n)
{
    return vfmulvf_float16xm1(va, a, n);
}

//
template <typename vT, int32_t v_len>
inline vT vaddvv(vT va, vT vb, uint64_t n);

template <>
inline float32xm1_t vaddvv<float32xm1_t, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vfaddvv_float32xm1(va, vb, n);
}
template <>
inline float16xm1_t vaddvv<float16xm1_t, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vfaddvv_float16xm1(va, vb, n);
}

}}}; // namespace ppl::kernel::riscv

#endif
