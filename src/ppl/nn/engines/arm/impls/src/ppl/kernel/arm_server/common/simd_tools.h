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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_SIMD_TOOLS_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_SIMD_TOOLS_H_

#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

inline void memcpy_neon(void *dst, const void *src, const uint64_t n)
{
    uint64_t __n         = n;
    uint8_t *__dst       = (uint8_t *)dst;
    const uint8_t *__src = (const uint8_t *)src;
    while (__n >= 32) {
        vst1q_u8(__dst + 0, vld1q_u8(__src + 0));
        vst1q_u8(__dst + 16, vld1q_u8(__src + 16));
        __dst += 32;
        __src += 32;
        __n -= 32;
    }
    if (__n & 16) {
        vst1q_u8(__dst + 0, vld1q_u8(__src + 0));
        __dst += 16;
        __src += 16;
    }
    if (__n & 8) {
        vst1_u8(__dst + 0, vld1_u8(__src + 0));
        __dst += 8;
        __src += 8;
    }
    if (__n & 4) {
        __dst[0] = __src[0];
        __dst[1] = __src[1];
        __dst[2] = __src[2];
        __dst[3] = __src[3];
        __dst += 4;
        __src += 4;
    }
    if (__n & 2) {
        __dst[0] = __src[0];
        __dst[1] = __src[1];
        __dst += 2;
        __src += 2;
    }
    if (__n & 1) {
        __dst[0] = __src[0];
    }
}

}}}; // namespace ppl::kernel::arm_server

#endif // __ST_PPL_KERNEL_ARM_SERVER_COMMON_SIMD_TOOLS_H_
