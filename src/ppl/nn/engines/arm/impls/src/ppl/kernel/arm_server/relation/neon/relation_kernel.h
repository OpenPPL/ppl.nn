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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_KERNEL_H_
#define __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_KERNEL_H_

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <>
inline uint8_t relation_scalar_kernel<float, RELATION_GREATER>(float a,float b)
{
    return a > b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<float, RELATION_GREATER_OR_EQUAL>(float a, float b)
{
    return a >= b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<float, RELATION_LESS>(float a, float b)
{
    return a < b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<float, RELATION_LESS_OR_EQUAL>(float a, float b)
{
    return a <= b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<float, RELATION_EQUAL>(float a, float b)
{
    return a == b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<float, RELATION_NOT_EQUAL>(float a, float b)
{
    return a != b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<int64_t, RELATION_GREATER>(int64_t a, int64_t b)
{
    return a > b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<int64_t, RELATION_GREATER_OR_EQUAL>(int64_t a, int64_t b)
{
    return a >= b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<int64_t, RELATION_LESS>(int64_t a, int64_t b)
{
    return a < b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<int64_t, RELATION_LESS_OR_EQUAL>(int64_t a, int64_t b)
{
    return a <= b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<int64_t, RELATION_EQUAL>(int64_t a, int64_t b)
{
    return a == b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<int64_t, RELATION_NOT_EQUAL>(int64_t a, int64_t b)
{
    return a != b ? 1 : 0;
}

#ifdef PPLNN_USE_ARMV8_2_FP16

template <>
inline uint8_t relation_scalar_kernel<__fp16, RELATION_GREATER>(__fp16 a, __fp16 b)
{
    return a > b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<__fp16, RELATION_GREATER_OR_EQUAL>(__fp16 a, __fp16 b)
{
    return a >= b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<__fp16, RELATION_LESS>(__fp16 a, __fp16 b)
{
    return a < b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<__fp16, RELATION_LESS_OR_EQUAL>(__fp16 a, __fp16 b)
{
    return a <= b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<__fp16, RELATION_EQUAL>(__fp16 a, __fp16 b)
{
    return a == b ? 1 : 0;
}

template <>
inline uint8_t relation_scalar_kernel<__fp16, RELATION_NOT_EQUAL>(__fp16 a, __fp16 b)
{
    return a != b ? 1 : 0;
}

#endif

template <>
inline float32x4_t relation_vector_kernel<float32x4_t, RELATION_GREATER>(const float32x4_t v0, const float32x4_t v1){
    return vcvtq_f32_u32(vcgtq_f32(v0, v1));
}

template <>
inline float32x4_t relation_vector_kernel<float32x4_t, RELATION_GREATER_OR_EQUAL>(const float32x4_t v0, const float32x4_t v1){
    return vcvtq_f32_u32(vcgeq_f32(v0, v1));
}


template <>
inline float32x4_t relation_vector_kernel<float32x4_t, RELATION_LESS>(const float32x4_t v0, const float32x4_t v1){
    return vcvtq_f32_u32(vcltq_f32(v0, v1));
}


template <>
inline float32x4_t relation_vector_kernel<float32x4_t, RELATION_LESS_OR_EQUAL>(const float32x4_t v0, const float32x4_t v1){
    return vcvtq_f32_u32(vcleq_f32(v0, v1));
}


template <>
inline float32x4_t relation_vector_kernel<float32x4_t, RELATION_EQUAL>(const float32x4_t v0, const float32x4_t v1){
    return vcvtq_f32_u32(vceqq_f32(v0, v1));
}

template <>
inline float32x4_t relation_vector_kernel<float32x4_t, RELATION_NOT_EQUAL>(const float32x4_t v0, const float32x4_t v1){
#ifdef __aarch64__
    return vcvtq_f32_u32(vceqzq_u32(vceqq_f32(v0, v1)));
#else
    uint32x4_t v_ceq = vceqq_f32(v0, v1);
    uint32x4_t v_ceqz = {0};
    vsetq_lane_u32((uint32_t)(vgetq_lane_u32(v_ceq, 0) == 0 ? std::numeric_limits<uint32_t>::max() : 0), v_ceqz, 0);
    vsetq_lane_u32((uint32_t)(vgetq_lane_u32(v_ceq, 1) == 0 ? std::numeric_limits<uint32_t>::max() : 0), v_ceqz, 1);
    vsetq_lane_u32((uint32_t)(vgetq_lane_u32(v_ceq, 2) == 0 ? std::numeric_limits<uint32_t>::max() : 0), v_ceqz, 2);
    vsetq_lane_u32((uint32_t)(vgetq_lane_u32(v_ceq, 3) == 0 ? std::numeric_limits<uint32_t>::max() : 0), v_ceqz, 3);
    return vcvtq_f32_u32(v_ceqz);
#endif
}

template <>
inline int64x2_t relation_vector_kernel<int64x2_t, RELATION_GREATER>(const int64x2_t v0, const int64x2_t v1){
    return vcgt<int64x2_t>(v0, v1);
}

template <>
inline int64x2_t relation_vector_kernel<int64x2_t, RELATION_GREATER_OR_EQUAL>(const int64x2_t v0, const int64x2_t v1){
    return vcge<int64x2_t>(v0, v1);
}


template <>
inline int64x2_t relation_vector_kernel<int64x2_t, RELATION_LESS>(const int64x2_t v0, const int64x2_t v1){
    return vclt<int64x2_t>(v0, v1);
}


template <>
inline int64x2_t relation_vector_kernel<int64x2_t, RELATION_LESS_OR_EQUAL>(const int64x2_t v0, const int64x2_t v1){
    return vcle<int64x2_t>(v0, v1);
}


template <>
inline int64x2_t relation_vector_kernel<int64x2_t, RELATION_EQUAL>(const int64x2_t v0, const int64x2_t v1){
    return vceq<int64x2_t>(v0, v1);
}

template <>
inline int64x2_t relation_vector_kernel<int64x2_t, RELATION_NOT_EQUAL>(const int64x2_t v0, const int64x2_t v1){
#ifdef __aarch64__
    return vreinterpretq_s64_u64(vceqzq_u64(vceqq_s64(v0, v1)));
#else
    int64x2_t v_ceq = vceq(v0, v1);
    uint64x2_t v_ceqz = {0};
    vsetq_lane_u64((uint32_t)(vgetq_lane_s64(v_ceq, 0) == 0 ? std::numeric_limits<uint64_t>::max() : 0), v_ceqz, 0);
    vsetq_lane_u64((uint32_t)(vgetq_lane_s64(v_ceq, 1) == 0 ? std::numeric_limits<uint64_t>::max() : 0), v_ceqz, 1);
    return vreinterpretq_s64_u64(v_ceqz);
#endif
}

#ifdef PPLNN_USE_ARMV8_2_FP16

template <>
inline float16x8_t relation_vector_kernel<float16x8_t, RELATION_GREATER>(const float16x8_t v0, const float16x8_t v1){
    return vcvtq_f16_u16(vcgtq_f16(v0, v1));
}

template <>
inline float16x8_t relation_vector_kernel<float16x8_t, RELATION_GREATER_OR_EQUAL>(const float16x8_t v0, const float16x8_t v1){
    return vcvtq_f16_u16(vcgeq_f16(v0, v1));
}


template <>
inline float16x8_t relation_vector_kernel<float16x8_t, RELATION_LESS>(const float16x8_t v0, const float16x8_t v1){
    return vcvtq_f16_u16(vcltq_f16(v0, v1));
}


template <>
inline float16x8_t relation_vector_kernel<float16x8_t, RELATION_LESS_OR_EQUAL>(const float16x8_t v0, const float16x8_t v1){
    return vcvtq_f16_u16(vcleq_f16(v0, v1));
}


template <>
inline float16x8_t relation_vector_kernel<float16x8_t, RELATION_EQUAL>(const float16x8_t v0, const float16x8_t v1){
    return vcvtq_f16_u16(vceqq_f16(v0, v1));
}

template <>
inline float16x8_t relation_vector_kernel<float16x8_t, RELATION_NOT_EQUAL>(const float16x8_t v0, const float16x8_t v1){
    return vcvtq_f16_u16(vceqzq_u16(vceqq_f16(v0, v1)));
}

#endif

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_KERNEL_H_
