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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_KERNEL_H_
#define __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_KERNEL_H_

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <>
inline float arithmetic_scalar_kernel<float, ARITHMETIC_ADD>(const float s0, const float s1)
{
    return s0 + s1;
}

template <>
inline float arithmetic_scalar_kernel<float, ARITHMETIC_SUB>(const float s0, const float s1)
{
    return s0 - s1;
}

template <>
inline float arithmetic_scalar_kernel<float, ARITHMETIC_MUL>(const float s0, const float s1)
{
    return s0 * s1;
}

template <>
inline float arithmetic_scalar_kernel<float, ARITHMETIC_DIV>(const float s0, const float s1)
{
    return s0 / s1;
}

template <>
inline float32x4_t arithmetic_vector_kernel<float32x4_t, ARITHMETIC_ADD>(const float32x4_t v0, const float32x4_t v1)
{
    return vadd(v0, v1);
}

template <>
inline float32x4_t arithmetic_vector_kernel<float32x4_t, ARITHMETIC_SUB>(const float32x4_t v0, const float32x4_t v1)
{
    return vsub(v0, v1);
}

template <>
inline float32x4_t arithmetic_vector_kernel<float32x4_t, ARITHMETIC_MUL>(const float32x4_t v0, const float32x4_t v1)
{
    return vmul(v0, v1);
}

template <>
inline float32x4_t arithmetic_vector_kernel<float32x4_t, ARITHMETIC_DIV>(const float32x4_t v0, const float32x4_t v1)
{
    return vdiv(v0, v1);
}

template <>
inline int64_t arithmetic_scalar_kernel<int64_t, ARITHMETIC_ADD>(const int64_t s0, const int64_t s1)
{
    return s0 + s1;
}

template <>
inline int64_t arithmetic_scalar_kernel<int64_t, ARITHMETIC_SUB>(const int64_t s0, const int64_t s1)
{
    return s0 - s1;
}

template <>
inline int64_t arithmetic_scalar_kernel<int64_t, ARITHMETIC_MUL>(const int64_t s0, const int64_t s1)
{
    return s0 * s1;
}

template <>
inline int64_t arithmetic_scalar_kernel<int64_t, ARITHMETIC_DIV>(const int64_t s0, const int64_t s1)
{
    return s0 / s1;
}

template <>
inline int64x2_t arithmetic_vector_kernel<int64x2_t, ARITHMETIC_ADD>(const int64x2_t v0, const int64x2_t v1)
{
    return vadd(v0, v1);
}

template <>
inline int64x2_t arithmetic_vector_kernel<int64x2_t, ARITHMETIC_SUB>(const int64x2_t v0, const int64x2_t v1)
{
    return vsub(v0, v1);
}

template <>
inline int64x2_t arithmetic_vector_kernel<int64x2_t, ARITHMETIC_MUL>(const int64x2_t v0, const int64x2_t v1)
{
    return vmul(v0, v1);
}

template <>
inline int64x2_t arithmetic_vector_kernel<int64x2_t, ARITHMETIC_DIV>(const int64x2_t v0, const int64x2_t v1)
{
    return vdiv(v0, v1);
}

#ifdef PPLNN_USE_ARMV8_2_FP16

template <>
inline __fp16 arithmetic_scalar_kernel<__fp16, ARITHMETIC_ADD>(const __fp16 s0, const __fp16 s1)
{
    return s0 + s1;
}

template <>
inline __fp16 arithmetic_scalar_kernel<__fp16, ARITHMETIC_SUB>(const __fp16 s0, const __fp16 s1)
{
    return s0 - s1;
}

template <>
inline __fp16 arithmetic_scalar_kernel<__fp16, ARITHMETIC_MUL>(const __fp16 s0, const __fp16 s1)
{
    return s0 * s1;
}

template <>
inline __fp16 arithmetic_scalar_kernel<__fp16, ARITHMETIC_DIV>(const __fp16 s0, const __fp16 s1)
{
    return s0 / s1;
}

template <>
inline float16x8_t arithmetic_vector_kernel<float16x8_t, ARITHMETIC_ADD>(const float16x8_t v0, const float16x8_t v1)
{
    return vadd(v0, v1);
}

template <>
inline float16x8_t arithmetic_vector_kernel<float16x8_t, ARITHMETIC_SUB>(const float16x8_t v0, const float16x8_t v1)
{
    return vsub(v0, v1);
}

template <>
inline float16x8_t arithmetic_vector_kernel<float16x8_t, ARITHMETIC_MUL>(const float16x8_t v0, const float16x8_t v1)
{
    return vmul(v0, v1);
}

template <>
inline float16x8_t arithmetic_vector_kernel<float16x8_t, ARITHMETIC_DIV>(const float16x8_t v0, const float16x8_t v1)
{
    return vdiv(v0, v1);
}

#endif

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_KERNEL_H_