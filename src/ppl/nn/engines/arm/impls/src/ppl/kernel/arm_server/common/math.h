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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_MATH_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_MATH_H_

#include <algorithm>
#include <type_traits>
#include <stdint.h>
#include <limits.h>
#include <limits>

namespace ppl { namespace kernel { namespace arm_server {

template <typename T>
inline T max(const T a, const T b)
{
    static_assert(std::is_fundamental<T>::value, "only allow fundamental type");
    return a > b ? a : b;
}

template <typename T>
inline T min(const T a, const T b)
{
    static_assert(std::is_fundamental<T>::value, "only allow fundamental type");
    return a < b ? a : b;
}

template <typename T>
inline T abs(const T x)
{
    static_assert(std::is_arithmetic<T>::value, "only allow arithmetic type");
    return x > 0 ? x : (-x);
}

template <typename T0, typename T1>
inline T0 div_up(const T0 a, const T1 b)
{
    static_assert(std::is_integral<T0>::value && std::is_integral<T1>::value, "only allow integral type");
    const T0 tb = static_cast<T0>(b);
    return (a + tb - static_cast<T0>(1)) / tb;
}

template <typename T0, typename T1>
inline T0 round(const T0 a, const T1 b)
{
    static_assert(std::is_integral<T0>::value && std::is_integral<T1>::value, "only allow integral type");
    const T0 tb = static_cast<T0>(b);
    return a / tb * tb;
}

template <typename T0, typename T1>
inline T0 round_up(const T0 a, const T1 b)
{
    static_assert(std::is_integral<T0>::value && std::is_integral<T1>::value, "only allow integral type");
    const T0 tb = static_cast<T0>(b);
    return (a + tb - static_cast<T0>(1)) / tb * tb;
}

template <typename T0, typename T1>
inline T0 mod_up(const T0 a, const T1 m)
{
    static_assert(std::is_integral<T0>::value && std::is_integral<T1>::value, "only allow integral type");
    const T0 tm = static_cast<T0>(m);
    return (a % tm == 0 && a != 0) ? tm : a % tm;
}

template <typename DataType, typename IndexType>
inline void argsort(const DataType *src, IndexType *indices, const int64_t length, const bool dec = true)
{
    for (int64_t i = 0; i < length; ++i)
        indices[i] = static_cast<IndexType>(i);
    if (dec) {
        std::stable_sort(indices, indices + length, [&src](const IndexType &ind0, const IndexType &ind1) {
            return src[ind0] > src[ind1];
        });
    } else {
        std::stable_sort(indices, indices + length, [&src](const IndexType &ind0, const IndexType &ind1) {
            return src[ind0] < src[ind1];
        });
    }
}

template <typename T>
inline T numeric_min(void)
{
    return std::numeric_limits<T>().min();
}

template <typename T>
inline T numeric_max(void)
{
    return std::numeric_limits<T>().max();
}

template <>
inline float numeric_min(void)
{
    return -numeric_max<float>();
}

template <>
inline double numeric_min(void)
{
    return -numeric_max<double>();
}

template <>
inline long double numeric_min(void)
{
    return -numeric_max<long double>();
}

// fp16 specialization

#ifdef PPLNN_USE_ARMV8_2_FP16
template <>
inline __fp16 max(const __fp16 a, const __fp16 b)
{
    return a > b ? a : b;
}

template <>
inline __fp16 min(const __fp16 a, const __fp16 b)
{
    return a < b ? a : b;
}

template <>
inline __fp16 numeric_max(void)
{
    return (__fp16)(6.5504e+4);
}

template <>
inline __fp16 numeric_min(void)
{
    return -numeric_max<__fp16>();
}

#endif

}}}; // namespace ppl::kernel::arm_server

#endif
