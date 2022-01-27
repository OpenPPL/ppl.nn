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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_KERNEL_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_KERNEL_H_

#include <riscv-vector.h>

namespace ppl { namespace kernel { namespace riscv {

enum relation_op_type_t {
    RELATION_GREATER          = 0,
    RELATION_GREATER_OR_EQUAL = 1,
    RELATION_LESS             = 2,
    RELATION_LESS_OR_EQUAL    = 3,
    RELATION_EQUAL            = 4,
    RELATION_NOT_EQUAL        = 5
};
//
template <typename T>
struct uint_type_helper;

template <>
struct uint_type_helper<float> {
    typedef uint32_t U;
};
template <>
struct uint_type_helper<__fp16> {
    typedef uint16_t U;
};
template <>
struct uint_type_helper<int64_t> {
    typedef uint64_t U;
};

template <typename T>
using uint_type = typename uint_type_helper<T>::U;

// ** 0. set vector-register **
// ** 0.1 **
template <typename T, int32_t vlen>
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
template <>
inline uint64_t vsetvli<int64_t, 128>(const int32_t c_blk)
{
    return vsetvli(c_blk, RVV_E64, RVV_M1);
}
// ** 0.2 **
template <typename T, int32_t vlen>
struct register_v_helper;

template <>
struct register_v_helper<float, 128> {
    typedef float32xm1_t U;
};
template <>
struct register_v_helper<__fp16, 128> {
    typedef float16xm1_t U;
};
template <>
struct register_v_helper<int64_t, 128> {
    typedef int64xm1_t U;
};
template <>
struct register_v_helper<uint32_t, 128> {
    typedef uint32xm1_t U;
};
template <>
struct register_v_helper<uint16_t, 128> {
    typedef uint16xm1_t U;
};
template <>
struct register_v_helper<uint64_t, 128> {
    typedef uint64xm1_t U;
};

template <typename T, int32_t vlen>
using register_v = typename register_v_helper<T, vlen>::U;
// ** 0.3 **
template <typename T, int32_t vlen>
struct register_ve_helper;

template <>
struct register_ve_helper<float, 128> {
    typedef e32xm1_t U;
};
template <>
struct register_ve_helper<__fp16, 128> {
    typedef e16xm1_t U;
};
template <>
struct register_ve_helper<int64_t, 128> {
    typedef e64xm1_t U;
};

template <typename T, int32_t vlen>
using register_ve = typename register_ve_helper<T, vlen>::U;

// ** 1. scalar kernel **
template <relation_op_type_t op, typename T>
inline uint8_t relation_scalar_kernel(T a, T b)
{
    if (RELATION_GREATER == op) return a > b ? 1 : 0;
    if (RELATION_GREATER_OR_EQUAL == op) return a >= b ? 1 : 0;
    if (RELATION_LESS == op) return a < b ? 1 : 0;
    if (RELATION_LESS_OR_EQUAL == op) return a <= b ? 1 : 0;
    if (RELATION_EQUAL == op) return a == b ? 1 : 0;
    if (RELATION_NOT_EQUAL == op) return a != b ? 1 : 0;
}

// ** 2. vector kernel **
// ** 2.1 vlev **
template <typename T, int32_t vlen>
inline register_v<T, vlen> vlev(const T* addr, uint64_t n);

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
template <>
inline int64xm1_t vlev<int64_t, 128>(const int64_t* addr, uint64_t n)
{
    return vlev_int64xm1(addr, n);
}
template <>
inline uint32xm1_t vlev<uint32_t, 128>(const uint32_t* addr, uint64_t n)
{
    return vlev_uint32xm1(addr, n);
}
template <>
inline uint16xm1_t vlev<uint16_t, 128>(const uint16_t* addr, uint64_t n)
{
    return vlev_uint16xm1(addr, n);
}
template <>
inline uint64xm1_t vlev<uint64_t, 128>(const uint64_t* addr, uint64_t n)
{
    return vlev_uint64xm1(addr, n);
}
// ** 2.2 vsev **
template <typename T, int32_t vlen>
inline void vsev(T* addr, register_v<T, vlen> va, uint64_t n);

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
template <>
inline void vsev<int64_t, 128>(int64_t* addr, int64xm1_t va, uint64_t n)
{
    return vsev_int64xm1(addr, va, n);
}
// ** 2.2.1 vsev_mask **
template <typename T, int32_t vlen>
inline void vsev_mask(uint_type<T>* addr, register_v<uint_type<T>, vlen> va, register_ve<T, vlen> mask, uint64_t n);

template <>
inline void vsev_mask<float, 128>(uint32_t* addr, uint32xm1_t va, e32xm1_t mask, uint64_t n)
{
    return vsev_mask_uint32xm1(addr, va, mask, n);
}
template <>
inline void vsev_mask<__fp16, 128>(uint16_t* addr, uint16xm1_t va, e16xm1_t mask, uint64_t n)
{
    return vsev_mask_uint16xm1(addr, va, mask, n);
}
template <>
inline void vsev_mask<int64_t, 128>(uint64_t* addr, uint64xm1_t va, e64xm1_t mask, uint64_t n)
{
    return vsev_mask_uint64xm1(addr, va, mask, n);
}
// ** 2.3 vmvvx **
template <typename T, int32_t vlen>
inline register_v<T, vlen> vmvvx(T a, uint64_t n);

template <>
inline float32xm1_t vmvvx<float, 128>(float a, uint64_t n)
{
    return vfmvvf_float32xm1(a, n);
}
template <>
inline float16xm1_t vmvvx<__fp16, 128>(__fp16 a, uint64_t n)
{
    return vfmvvf_float16xm1(a, n);
}
template <>
inline int64xm1_t vmvvx<int64_t, 128>(int64_t a, uint64_t n)
{
    return vmvvx_int64xm1(a, n);
}

// ** 2.4 opetaion **
template <relation_op_type_t op, typename T, int32_t vlen>
inline register_ve<T, vlen> vrelation_vv(register_v<T, vlen> va, register_v<T, vlen> vb, uint64_t n);
// ** 2.4.1 greater **
template <>
inline e32xm1_t vrelation_vv<RELATION_GREATER, float, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vmfltvv_e32xm1_float32xm1(vb, va, n);
}
template <>
inline e16xm1_t vrelation_vv<RELATION_GREATER, __fp16, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vmfltvv_e16xm1_float16xm1(vb, va, n);
}
template <>
inline e64xm1_t vrelation_vv<RELATION_GREATER, int64_t, 128>(int64xm1_t va, int64xm1_t vb, uint64_t n)
{
    return vmsltvv_e64xm1_int64xm1(vb, va, n);
}
// ** 2.4.2 greater_or_equal **
template <>
inline e32xm1_t vrelation_vv<RELATION_GREATER_OR_EQUAL, float, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vmflevv_e32xm1_float32xm1(vb, va, n);
}
template <>
inline e16xm1_t vrelation_vv<RELATION_GREATER_OR_EQUAL, __fp16, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vmflevv_e16xm1_float16xm1(vb, va, n);
}
template <>
inline e64xm1_t vrelation_vv<RELATION_GREATER_OR_EQUAL, int64_t, 128>(int64xm1_t va, int64xm1_t vb, uint64_t n)
{
    return vmslevv_e64xm1_int64xm1(vb, va, n);
}
// ** 2.4.3 less **
template <>
inline e32xm1_t vrelation_vv<RELATION_LESS, float, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vmfltvv_e32xm1_float32xm1(va, vb, n);
}
template <>
inline e16xm1_t vrelation_vv<RELATION_LESS, __fp16, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vmfltvv_e16xm1_float16xm1(va, vb, n);
}
template <>
inline e64xm1_t vrelation_vv<RELATION_LESS, int64_t, 128>(int64xm1_t va, int64xm1_t vb, uint64_t n)
{
    return vmsltvv_e64xm1_int64xm1(va, vb, n);
}
// ** 2.4.4 less_or_equal **
template <>
inline e32xm1_t vrelation_vv<RELATION_LESS_OR_EQUAL, float, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vmflevv_e32xm1_float32xm1(va, vb, n);
}
template <>
inline e16xm1_t vrelation_vv<RELATION_LESS_OR_EQUAL, __fp16, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vmflevv_e16xm1_float16xm1(va, vb, n);
}
template <>
inline e64xm1_t vrelation_vv<RELATION_LESS_OR_EQUAL, int64_t, 128>(int64xm1_t va, int64xm1_t vb, uint64_t n)
{
    return vmslevv_e64xm1_int64xm1(va, vb, n);
}
// ** 2.4.5 equal **
template <>
inline e32xm1_t vrelation_vv<RELATION_EQUAL, float, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vmfeqvv_e32xm1_float32xm1(va, vb, n);
}
template <>
inline e16xm1_t vrelation_vv<RELATION_EQUAL, __fp16, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vmfeqvv_e16xm1_float16xm1(va, vb, n);
}
template <>
inline e64xm1_t vrelation_vv<RELATION_EQUAL, int64_t, 128>(int64xm1_t va, int64xm1_t vb, uint64_t n)
{
    return vmseqvv_e64xm1_int64xm1(va, vb, n);
}
// ** 2.4.6 not equal **
template <>
inline e32xm1_t vrelation_vv<RELATION_NOT_EQUAL, float, 128>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vmfnevv_e32xm1_float32xm1(va, vb, n);
}
template <>
inline e16xm1_t vrelation_vv<RELATION_NOT_EQUAL, __fp16, 128>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vmfnevv_e16xm1_float16xm1(va, vb, n);
}
template <>
inline e64xm1_t vrelation_vv<RELATION_NOT_EQUAL, int64_t, 128>(int64xm1_t va, int64xm1_t vb, uint64_t n)
{
    return vmsnevv_e64xm1_int64xm1(va, vb, n);
}
}}}; // namespace ppl::kernel::riscv

#endif
