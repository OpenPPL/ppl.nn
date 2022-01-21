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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_MATH_NEON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_MATH_NEON_H_

#include <arm_neon.h>

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

inline float32x4_t v_exp_f32(const float32x4_t v_src)
{
    float32x4_t tmp = vdupq_n_f32(0.0f);
    float32x4_t one  = vdupq_n_f32(1.0f);
    float32x4_t x    = v_src;
    x  = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
    x  = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));
    float32x4_t fx = vfmaq_f32(x, vdupq_n_f32(1.44269504088896341), vdupq_n_f32(0.5f));

    tmp = vrndmq_f32(fx);
    //TODO: compare is right?
    uint32x4_t mask = vceqq_f32(tmp, fx);
    mask = vandq_u32(mask, vcvtq_u32_f32(one));
    fx = vsubq_f32(tmp, vcvtq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(0.693359375));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(-2.12194440e-4));
    x   = vsubq_f32(x, tmp);
    x   = vsubq_f32(x, z);
    z   = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(1.9875691500E-4);
    y = vfmaq_f32(y, x, vdupq_n_f32(1.3981999507E-3));
    y = vfmaq_f32(y, x, vdupq_n_f32(8.3334519073E-3));
    y = vfmaq_f32(y, x, vdupq_n_f32(4.1665795894E-2));
    y = vfmaq_f32(y, x, vdupq_n_f32(1.6666665459E-1));
    y = vfmaq_f32(y, x, vdupq_n_f32(5.0000001201E-1));
    y = vfmaq_f32(y, z, x);
    y = vaddq_f32(y, one);
    
    int32x4_t imm0 = vcvtq_s32_f32(fx);
    imm0 = vaddq_s32(imm0, vdupq_n_s32(0x7f));
    imm0 = vqrshlq_s32(imm0, vdupq_n_s32(23));
    float32x4_t pow2n = vcvtq_f32_s32(imm0);
    y = vmulq_f32(y, pow2n);

    return y;
}


inline float32x4_t v_sigmoid_f32(const float32x4_t v_src)
{
    float32x4_t value = v_src;
    value             = vmaxq_f32(vdupq_n_f32(-18.0f), value);
    value             = vminq_f32(vdupq_n_f32(18.0f), value);

    float32x4_t value_squared = vmulq_f32(value, value);

    float32x4_t p;
    p = vfmaq_f32(vdupq_n_f32(1.15627324459942e-07f), vdupq_n_f32(4.37031012579801e-11f), value_squared);
    p = vfmaq_f32(vdupq_n_f32(6.08574864600143e-05f), p, value_squared);
    p = vfmaq_f32(vdupq_n_f32(8.51377133304701e-03f), p, value_squared);
    p = vfmaq_f32(vdupq_n_f32(2.48287947061529e-01f), p, value_squared);
    p = vmulq_f32(p, value);

    float32x4_t q;
    q = vfmaq_f32(vdupq_n_f32(5.76102136993427e-09f), vdupq_n_f32(6.10247389755681e-13f), value_squared);
    q = vfmaq_f32(vdupq_n_f32(6.29106785017040e-06f), q, value_squared);
    q = vfmaq_f32(vdupq_n_f32(1.70198817374094e-03f), q, value_squared);
    q = vfmaq_f32(vdupq_n_f32(1.16817656904453e-01f), q, value_squared);
    q = vfmaq_f32(vdupq_n_f32(9.93151921023180e-01f), q, value_squared);

    float32x4_t dst = vaddq_f32(vdivq_f32(p, q), vdupq_n_f32(0.5f));
    return dst;
}

}}}} // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_COMMON_MATH_NEON_H_

