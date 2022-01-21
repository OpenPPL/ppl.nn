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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_TYPE_TRAITS_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_TYPE_TRAITS_H_

#include <stdint.h>
#include <math.h>
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

/********************** interface **********************/

template <typename eT, int32_t eN>
struct DT;

template <typename eT, int32_t eN>
inline typename DT<eT, eN>::vecDT vld(const eT* ptr);

template <typename eT, int32_t eN>
inline void vst(eT* ptr, const typename DT<eT, eN>::vecDT v);

template <typename eT, int32_t eN>
inline typename DT<eT, eN>::vecDT vdup_n(const eT s);

template <typename vecType>
inline vecType vadd(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vsub(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vmul(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vdiv(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vmin(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vmax(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vzip1(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vzip2(const vecType& v0, const vecType& v1);

template <typename vecType>
inline vecType vsqrt(const vecType& v0);

template <typename vecType>
inline vecType operator+(const vecType& v0, const vecType& v1)
{
    return vadd<vecType>(v0, v1);
}

template <typename vecType>
inline vecType operator-(const vecType& v0, const vecType& v1)
{
    return vsub<vecType>(v0, v1);
}

template <typename vecType>
inline vecType operator*(const vecType& v0, const vecType& v1)
{
    return vmul<vecType>(v0, v1);
}

template <typename vecType>
inline vecType operator/(const vecType& v0, const vecType& v1)
{
    return vdiv<vecType>(v0, v1);
}

/********************** fp32 x 4 **********************/

template <>
struct DT<float, 4> {
    typedef float32x4_t vecDT;
};

template <>
inline float32x4_t vld<float, 4>(const float* ptr)
{
    return vld1q_f32(ptr);
}

template <>
inline void vst<float, 4>(float* ptr, const float32x4_t v)
{
    vst1q_f32(ptr, v);
}

template <>
inline float32x4_t vdup_n<float, 4>(const float s)
{
    return vdupq_n_f32(s);
}

template <>
inline float32x4_t vadd(const float32x4_t& v0, const float32x4_t& v1)
{
    return vaddq_f32(v0, v1);
}

template <>
inline float32x4_t vsub(const float32x4_t& v0, const float32x4_t& v1)
{
    return vsubq_f32(v0, v1);
}

template <>
inline float32x4_t vmul(const float32x4_t& v0, const float32x4_t& v1)
{
    return vmulq_f32(v0, v1);
}

template <>
inline float32x4_t vdiv(const float32x4_t& v0, const float32x4_t& v1)
{
    return vdivq_f32(v0, v1);
}

template <>
inline float32x4_t vmin(const float32x4_t& v0, const float32x4_t& v1)
{
    return vminq_f32(v0, v1);
}

template <>
inline float32x4_t vmax(const float32x4_t& v0, const float32x4_t& v1)
{
    return vmaxq_f32(v0, v1);
}

template <>
inline float32x4_t vzip1<float32x4_t>(const float32x4_t& v0, const float32x4_t& v1)
{
    return vzip1q_f32(v0, v1);
}

template <>
inline float32x4_t vzip2<float32x4_t>(const float32x4_t& v0, const float32x4_t& v1)
{
    return vzip2q_f32(v0, v1);
}

template <>
inline float32x4_t vsqrt<float32x4_t>(const float32x4_t& v0)
{
    return vsqrtq_f32(v0);
}

/********************** uint32 x 2 **********************/

template <>
struct DT<uint32_t, 2> {
    typedef uint32x2_t vecDT;
};

template <>
inline uint32x2_t vld<uint32_t, 2>(const uint32_t* ptr)
{
    return vld1_u32(ptr);
}

template <>
inline void vst<uint32_t, 2>(uint32_t* ptr, const uint32x2_t v)
{
    vst1_u32(ptr, v);
}

template <>
inline uint32x2_t vdup_n<uint32_t, 2>(const uint32_t s)
{
    return vdup_n_u32(s);
}

template <>
inline uint32x2_t vadd(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vadd_u32(v0, v1);
}

template <>
inline uint32x2_t vsub(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vsub_u32(v0, v1);
}

template <>
inline uint32x2_t vmul(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vmul_u32(v0, v1);
}

template <>
inline uint32x2_t vdiv(const uint32x2_t& v0, const uint32x2_t& v1)
{
    uint32x2_t v_dst = {0};
    vset_lane_u32(vget_lane_u32(v0, 0) / vget_lane_u32(v1, 0), v_dst, 0);
    vset_lane_u32(vget_lane_u32(v0, 1) / vget_lane_u32(v1, 1), v_dst, 1);
    return v_dst;
}

template <>
inline uint32x2_t vmin(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vmin_u32(v0, v1);
}

template <>
inline uint32x2_t vmax(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vmax_u32(v0, v1);
}

template <>
inline uint32x2_t vzip1<uint32x2_t>(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vzip1_u32(v0, v1);
}

template <>
inline uint32x2_t vzip2<uint32x2_t>(const uint32x2_t& v0, const uint32x2_t& v1)
{
    return vzip2_u32(v0, v1);
}

template <>
inline uint32x2_t vsqrt<uint32x2_t>(const uint32x2_t& v0)
{
    uint32x2_t v_dst = {0};
    vset_lane_u32((uint32_t)sqrtf(vget_lane_u32(v0, 0)), v_dst, 0);
    vset_lane_u32((uint32_t)sqrtf(vget_lane_u32(v0, 1)), v_dst, 1);
    return v_dst;
}

/********************** uint32 x 4 **********************/

template <>
struct DT<uint32_t, 4> {
    typedef uint32x4_t vecDT;
};

template <>
inline uint32x4_t vld<uint32_t, 4>(const uint32_t* ptr)
{
    return vld1q_u32(ptr);
}

template <>
inline void vst<uint32_t, 4>(uint32_t* ptr, const uint32x4_t v)
{
    vst1q_u32(ptr, v);
}

template <>
inline uint32x4_t vdup_n<uint32_t, 4>(const uint32_t s)
{
    return vdupq_n_u32(s);
}

template <>
inline uint32x4_t vadd(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vaddq_u32(v0, v1);
}

template <>
inline uint32x4_t vsub(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vsubq_u32(v0, v1);
}

template <>
inline uint32x4_t vmul(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vmulq_u32(v0, v1);
}

template <>
inline uint32x4_t vdiv(const uint32x4_t& v0, const uint32x4_t& v1)
{
    uint32x4_t v_dst = {0};
    vsetq_lane_u32(vgetq_lane_u32(v0, 0) / vgetq_lane_u32(v1, 0), v_dst, 0);
    vsetq_lane_u32(vgetq_lane_u32(v0, 1) / vgetq_lane_u32(v1, 1), v_dst, 1);
    vsetq_lane_u32(vgetq_lane_u32(v0, 2) / vgetq_lane_u32(v1, 2), v_dst, 2);
    vsetq_lane_u32(vgetq_lane_u32(v0, 3) / vgetq_lane_u32(v1, 3), v_dst, 3);
    return v_dst;
}

template <>
inline uint32x4_t vmin(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vminq_u32(v0, v1);
}

template <>
inline uint32x4_t vmax(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vmaxq_u32(v0, v1);
}

template <>
inline uint32x4_t vzip1<uint32x4_t>(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vzip1q_u32(v0, v1);
}

template <>
inline uint32x4_t vzip2<uint32x4_t>(const uint32x4_t& v0, const uint32x4_t& v1)
{
    return vzip2q_u32(v0, v1);
}

template <>
inline uint32x4_t vsqrt<uint32x4_t>(const uint32x4_t& v0)
{
    uint32x4_t v_dst = {0};
    vsetq_lane_u32((uint32_t)sqrtf(vgetq_lane_u32(v0, 0)), v_dst, 0);
    vsetq_lane_u32((uint32_t)sqrtf(vgetq_lane_u32(v0, 1)), v_dst, 1);
    vsetq_lane_u32((uint32_t)sqrtf(vgetq_lane_u32(v0, 2)), v_dst, 2);
    vsetq_lane_u32((uint32_t)sqrtf(vgetq_lane_u32(v0, 3)), v_dst, 3);
    return v_dst;
}

/********************** int64 x 2 **********************/

template <>
struct DT<int64_t, 2> {
    typedef int64x2_t vecDT;
};

template <>
inline int64x2_t vld<int64_t, 2>(const int64_t* ptr)
{
    return vld1q_s64(ptr);
}

template <>
inline void vst<int64_t, 2>(int64_t* ptr, const int64x2_t v)
{
    vst1q_s64(ptr, v);
}

template <>
inline int64x2_t vdup_n<int64_t, 2>(const int64_t s)
{
    return vdupq_n_s64(s);
}

template <>
inline int64x2_t vadd(const int64x2_t& v0, const int64x2_t& v1)
{
    return vaddq_s64(v0, v1);
}

template <>
inline int64x2_t vsub(const int64x2_t& v0, const int64x2_t& v1)
{
    return vsubq_s64(v0, v1);
}

template <>
inline int64x2_t vmul(const int64x2_t& v0, const int64x2_t& v1)
{
    int64x2_t v_dst = {0};
    vsetq_lane_s64(vgetq_lane_s64(v0, 0) * vgetq_lane_s64(v1, 0), v_dst, 0);
    vsetq_lane_s64(vgetq_lane_s64(v0, 1) * vgetq_lane_s64(v1, 1), v_dst, 1);
    return v_dst;
}

template <>
inline int64x2_t vdiv(const int64x2_t& v0, const int64x2_t& v1)
{
    int64x2_t v_dst = {0};
    vsetq_lane_s64(vgetq_lane_s64(v0, 0) / vgetq_lane_s64(v1, 0), v_dst, 0);
    vsetq_lane_s64(vgetq_lane_s64(v0, 1) / vgetq_lane_s64(v1, 1), v_dst, 1);
    return v_dst;
}

template <>
inline int64x2_t vmin(const int64x2_t& v0, const int64x2_t& v1)
{
    int64x2_t v_dst = {0};
    vsetq_lane_s64(min(vgetq_lane_s64(v0, 0), vgetq_lane_s64(v1, 0)), v_dst, 0);
    vsetq_lane_s64(min(vgetq_lane_s64(v0, 1), vgetq_lane_s64(v1, 1)), v_dst, 1);
    return v_dst;
}

template <>
inline int64x2_t vmax(const int64x2_t& v0, const int64x2_t& v1)
{
    int64x2_t v_dst = {0};
    vsetq_lane_s64(max(vgetq_lane_s64(v0, 0), vgetq_lane_s64(v1, 0)), v_dst, 0);
    vsetq_lane_s64(max(vgetq_lane_s64(v0, 1), vgetq_lane_s64(v1, 1)), v_dst, 1);
    return v_dst;
}

template <>
inline int64x2_t vzip1<int64x2_t>(const int64x2_t& v0, const int64x2_t& v1)
{
    return vzip1q_s64(v0, v1);
}

template <>
inline int64x2_t vzip2<int64x2_t>(const int64x2_t& v0, const int64x2_t& v1)
{
    return vzip2q_s64(v0, v1);
}

template <>
inline int64x2_t vsqrt<int64x2_t>(const int64x2_t& v0)
{
    int64x2_t v_dst = {0};
    vsetq_lane_s64((int64_t)sqrtf(vgetq_lane_s64(v0, 0)), v_dst, 0);
    vsetq_lane_s64((int64_t)sqrtf(vgetq_lane_s64(v0, 1)), v_dst, 1);
    return v_dst;
}

/********************** uint16 x 8 **********************/

template <>
struct DT<uint16_t, 8> {
    typedef uint16x8_t vecDT;
};

template <>
inline uint16x8_t vld<uint16_t, 8>(const uint16_t* ptr)
{
    return vld1q_u16(ptr);
}

template <>
inline void vst<uint16_t, 8>(uint16_t* ptr, const uint16x8_t v)
{
    vst1q_u16(ptr, v);
}

template <>
inline uint16x8_t vdup_n<uint16_t, 8>(const uint16_t s)
{
    return vdupq_n_u16(s);
}

template <>
inline uint16x8_t vadd(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vaddq_u16(v0, v1);
}

template <>
inline uint16x8_t vsub(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vsubq_u16(v0, v1);
}

template <>
inline uint16x8_t vmul(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vmulq_u16(v0, v1);
}

template <>
inline uint16x8_t vdiv(const uint16x8_t& v0, const uint16x8_t& v1)
{
    uint16x8_t v_dst = {0};
    vsetq_lane_u16(vgetq_lane_u16(v0, 0) / vgetq_lane_u16(v1, 0), v_dst, 0);
    vsetq_lane_u16(vgetq_lane_u16(v0, 1) / vgetq_lane_u16(v1, 1), v_dst, 1);
    vsetq_lane_u16(vgetq_lane_u16(v0, 2) / vgetq_lane_u16(v1, 2), v_dst, 2);
    vsetq_lane_u16(vgetq_lane_u16(v0, 3) / vgetq_lane_u16(v1, 3), v_dst, 3);
    vsetq_lane_u16(vgetq_lane_u16(v0, 4) / vgetq_lane_u16(v1, 4), v_dst, 4);
    vsetq_lane_u16(vgetq_lane_u16(v0, 5) / vgetq_lane_u16(v1, 5), v_dst, 5);
    vsetq_lane_u16(vgetq_lane_u16(v0, 6) / vgetq_lane_u16(v1, 6), v_dst, 6);
    vsetq_lane_u16(vgetq_lane_u16(v0, 7) / vgetq_lane_u16(v1, 7), v_dst, 7);
    return v_dst;
}

template <>
inline uint16x8_t vmin(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vminq_u16(v0, v1);
}

template <>
inline uint16x8_t vmax(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vmaxq_u16(v0, v1);
}

template <>
inline uint16x8_t vzip1<uint16x8_t>(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vzip1q_u16(v0, v1);
}

template <>
inline uint16x8_t vzip2<uint16x8_t>(const uint16x8_t& v0, const uint16x8_t& v1)
{
    return vzip2q_u16(v0, v1);
}

template <>
inline uint16x8_t vsqrt(const uint16x8_t& v0)
{
    uint16x8_t v_dst = {0};
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 0)), v_dst, 0);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 1)), v_dst, 1);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 2)), v_dst, 2);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 3)), v_dst, 3);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 4)), v_dst, 4);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 5)), v_dst, 5);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 6)), v_dst, 6);
    vsetq_lane_u16((uint16_t)sqrtf(vgetq_lane_u16(v0, 7)), v_dst, 7);
    return v_dst;
}

/********************** uint16 x 4 **********************/

template <>
struct DT<uint16_t, 4> {
    typedef uint16x4_t vecDT;
};

template <>
inline uint16x4_t vld<uint16_t, 4>(const uint16_t* ptr)
{
    return vld1_u16(ptr);
}

template <>
inline void vst<uint16_t, 4>(uint16_t* ptr, const uint16x4_t v)
{
    vst1_u16(ptr, v);
}

template <>
inline uint16x4_t vdup_n<uint16_t, 4>(const uint16_t s)
{
    return vdup_n_u16(s);
}

template <>
inline uint16x4_t vadd(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vadd_u16(v0, v1);
}

template <>
inline uint16x4_t vsub(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vsub_u16(v0, v1);
}

template <>
inline uint16x4_t vmul(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vmul_u16(v0, v1);
}

template <>
inline uint16x4_t vdiv(const uint16x4_t& v0, const uint16x4_t& v1)
{
    uint16x4_t v_dst = {0};
    vset_lane_u16(vget_lane_u16(v0, 0) / vget_lane_u16(v1, 0), v_dst, 0);
    vset_lane_u16(vget_lane_u16(v0, 1) / vget_lane_u16(v1, 1), v_dst, 1);
    vset_lane_u16(vget_lane_u16(v0, 2) / vget_lane_u16(v1, 2), v_dst, 2);
    vset_lane_u16(vget_lane_u16(v0, 3) / vget_lane_u16(v1, 3), v_dst, 3);
    return v_dst;
}

template <>
inline uint16x4_t vmin(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vmin_u16(v0, v1);
}

template <>
inline uint16x4_t vmax(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vmax_u16(v0, v1);
}

template <>
inline uint16x4_t vzip1<uint16x4_t>(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vzip1_u16(v0, v1);
}

template <>
inline uint16x4_t vzip2<uint16x4_t>(const uint16x4_t& v0, const uint16x4_t& v1)
{
    return vzip2_u16(v0, v1);
}

template <>
inline uint16x4_t vsqrt(const uint16x4_t& v0)
{
    uint16x4_t v_dst = {0};
    vset_lane_u16((uint16_t)sqrtf(vget_lane_u16(v0, 0)), v_dst, 0);
    vset_lane_u16((uint16_t)sqrtf(vget_lane_u16(v0, 1)), v_dst, 1);
    vset_lane_u16((uint16_t)sqrtf(vget_lane_u16(v0, 2)), v_dst, 2);
    vset_lane_u16((uint16_t)sqrtf(vget_lane_u16(v0, 3)), v_dst, 3);
    return v_dst;
}

/********************** fp16 x 8 **********************/

#ifdef PPL_USE_ARM_SERVER_FP16
template <>
struct DT<__fp16, 8> {
    typedef float16x8_t vecDT;
};

template <>
inline float16x8_t vld<__fp16, 8>(const __fp16* ptr)
{
    return vld1q_f16(ptr);
}

template <>
inline void vst<__fp16, 8>(__fp16* ptr, const float16x8_t v)
{
    vst1q_f16(ptr, v);
}

template <>
inline float16x8_t vdup_n<__fp16, 8>(const __fp16 s)
{
    return vdupq_n_f16(s);
}

template <>
inline float16x8_t vadd(const float16x8_t& v0, const float16x8_t& v1)
{
    return vaddq_f16(v0, v1);
}

template <>
inline float16x8_t vsub(const float16x8_t& v0, const float16x8_t& v1)
{
    return vsubq_f16(v0, v1);
}

template <>
inline float16x8_t vmul(const float16x8_t& v0, const float16x8_t& v1)
{
    return vmulq_f16(v0, v1);
}

template <>
inline float16x8_t vdiv(const float16x8_t& v0, const float16x8_t& v1)
{
    return vdivq_f16(v0, v1);
}

template <>
inline float16x8_t vmin(const float16x8_t& v0, const float16x8_t& v1)
{
    return vminq_f16(v0, v1);
}

template <>
inline float16x8_t vmax(const float16x8_t& v0, const float16x8_t& v1)
{
    return vmaxq_f16(v0, v1);
}

template <>
inline float16x8_t vzip1<float16x8_t>(const float16x8_t& v0, const float16x8_t& v1)
{
    return vzip1q_f16(v0, v1);
}

template <>
inline float16x8_t vzip2<float16x8_t>(const float16x8_t& v0, const float16x8_t& v1)
{
    return vzip2q_f16(v0, v1);
}

template <>
inline float16x8_t vsqrt<float16x8_t>(const float16x8_t& v0)
{
    return vsqrtq_f16(v0);
}
#endif

}}}; // namespace ppl::kernel::arm_server

#endif // __ST_PPL_KERNEL_ARM_SERVER_COMMON_TYPE_TRAITS_H_