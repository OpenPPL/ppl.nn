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

#ifndef __ST_PPL_KERNEL_X86_COMMON_MATH_SSE_H_
#define __ST_PPL_KERNEL_X86_COMMON_MATH_SSE_H_

#include <xmmintrin.h>

namespace ppl { namespace kernel { namespace x86 {

// an approximation of sigmoid
static inline __m128 _sse_sigmoid_ps(__m128 value)
{
    value = _mm_max_ps(_mm_set1_ps(-18.0f), value);
    value = _mm_min_ps(_mm_set1_ps(18.0f), value);

    __m128 value_squared = _mm_mul_ps(value, value);

    __m128 p;
    p = _mm_mul_ps(value_squared, _mm_set1_ps(4.37031012579801e-11f));
    p = _mm_add_ps(p, _mm_set1_ps(1.15627324459942e-07f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(6.08574864600143e-05f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(8.51377133304701e-03f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(2.48287947061529e-01f));
    p = _mm_mul_ps(p, value);

    __m128 q;
    q = _mm_mul_ps(value_squared, _mm_set1_ps(6.10247389755681e-13f));
    q = _mm_add_ps(q, _mm_set1_ps(5.76102136993427e-09f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(6.29106785017040e-06f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(1.70198817374094e-03f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(1.16817656904453e-01f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(9.93151921023180e-01f));

    __m128 dst = _mm_add_ps(_mm_div_ps(p, q), _mm_set1_ps(0.5f));
    return dst;
}

// an approximation of tanh
static inline __m128 _sse_tanh_ps(__m128 value)
{
    value = _mm_max_ps(_mm_set1_ps(-9.0f), value);
    value = _mm_min_ps(_mm_set1_ps(9.0f), value);

    __m128 value_squared = _mm_mul_ps(value, value);

    __m128 p;
    p = _mm_mul_ps(value_squared, _mm_set1_ps(-2.76076847742355e-16f));
    p = _mm_add_ps(p, _mm_set1_ps(2.00018790482477e-13f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(-8.60467152213735e-11f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(5.12229709037114e-08f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(1.48572235717979e-05f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(6.37261928875436e-04f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(4.89352455891786e-03f));
    p = _mm_mul_ps(p, value);

    __m128 q;
    q = _mm_mul_ps(value_squared, _mm_set1_ps(1.19825839466702e-06f));
    q = _mm_add_ps(q, _mm_set1_ps(1.18534705686654e-04f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(2.26843463243900e-03f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(4.89352518554385e-03f));

    __m128 dst = _mm_div_ps(p, q);
    return dst;
}

// an approximation of exp
static inline __m128 _sse_exp_ps(__m128 x)
{
    __m128 tmp = _mm_setzero_ps(), fx;
    __m128i imm0;
    __m128 one = _mm_set1_ps(1.0f);

    x = _mm_min_ps(x, _mm_set1_ps(88.3762626647949f));
    x = _mm_max_ps(x, _mm_set1_ps(-88.3762626647949f));

    fx = _mm_mul_ps(x, _mm_set1_ps(1.44269504088896341));
    fx = _mm_add_ps(fx, _mm_set1_ps(0.5f));

    tmp = _mm_floor_ps(fx);

    __m128 mask = _mm_cmpgt_ps(tmp, fx);
    mask        = _mm_and_ps(mask, one);
    fx          = _mm_sub_ps(tmp, mask);

    tmp      = _mm_mul_ps(fx, _mm_set1_ps(0.693359375));
    __m128 z = _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4));
    x        = _mm_sub_ps(x, tmp);
    x        = _mm_sub_ps(x, z);
    z        = _mm_mul_ps(x, x);

    __m128 y = _mm_set1_ps(1.9875691500E-4);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(1.3981999507E-3));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(8.3334519073E-3));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(4.1665795894E-2));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(1.6666665459E-1));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(5.0000001201E-1));
    y        = _mm_mul_ps(y, z);
    y        = _mm_add_ps(y, x);
    y        = _mm_add_ps(y, one);

    imm0         = _mm_cvttps_epi32(fx);
    imm0         = _mm_add_epi32(imm0, _mm_set1_epi32(0x7f));
    imm0         = _mm_slli_epi32(imm0, 23);
    __m128 pow2n = _mm_castsi128_ps(imm0);
    y            = _mm_mul_ps(y, pow2n);
    return y;
}

}}}; // namespace ppl::kernel::x86

#endif
