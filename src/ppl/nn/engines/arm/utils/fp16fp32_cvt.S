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

#ifndef __LP64__
.syntax unified
.fpu neon-fp16

.align 4

.global CvtFp32ToFp16
.hidden CvtFp32ToFp16
.type CvtFp32ToFp16, %function
CvtFp32ToFp16:
// param r0: counts
// param r1: src
// param r2: dst
.fp32.fp16.loop:
    vld1.32 {d0-d3}, [r1]!
    vcvt.f16.f32 d4, q0
    subs r0, #8
    vcvt.f16.f32 d5, q1
    vst1.16 {d4-d5}, [r2]!
    bne .fp32.fp16.loop
    bx lr

.global CvtFp16ToFp32
.hidden CvtFp16ToFp32
.type CvtFp16ToFp32, %function
CvtFp16ToFp32:
// param r0: counts
// param r1: src
// param r2: dst
.fp16.fp32.loop:
    vld1.16 {d0-d1}, [r1]!
    vcvt.f32.f16 q1, d0
    subs r0, #8
    vcvt.f32.f16 q2, d1
    vst1.32 {d2-d5}, [r2]!
    bne .fp16.fp32.loop
    bx lr

#else // not lp64
.align 2

.global CvtFp32ToFp16
.hidden CvtFp32ToFp16
.type CvtFp32ToFp16, %function
CvtFp32ToFp16:
// param x0: counts
// param x1: src
// param x2: dst
.fp32.fp16.loop:
    ld1 {v0.4s, v1.4s}, [x1], #32
    fcvtn v2.4h, v0.4s
    subs x0, x0, #8
    fcvtn2 v2.8h, v1.4s
    st1 {v2.8h}, [x2], #16
    bne .fp32.fp16.loop
    ret

.global CvtFp16ToFp32
.hidden CvtFp16ToFp32
.type CvtFp16ToFp32, %function
CvtFp16ToFp32:
// param x1: src
// param x2: dst
.fp16.fp32.loop:
    ld1 {v0.8h}, [x1], #16
    fcvtl v1.4s, v0.4h
    subs x0, x0, #8
    fcvtl2 v2.4s, v0.8h
    st1 {v1.4s, v2.4s}, [x2], #32
    bne .fp16.fp32.loop
    ret

#endif //! aarch64
