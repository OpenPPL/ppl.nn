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

#include <cstdint>

template <int64_t atom_w>
void conv_dw_f3s1_h1w4_kernel_riscv_fp32(
    const float* src,
    const float* flt,
    const float* bias,
    float* dst,

    int64_t src_pad_w,
    int64_t dst_h,
    int64_t dst_w)
{
    asm volatile(
        ".equ           ATOM_W, %c[ATOM_W]      \n\t"

        "addi           t0,     zero,   4       \n\t"
        "vsetvli        t1,     t0,     e32     \n\t"

        "mv             t0,     %[SRC]          \n\t"
        "mv             t1,     %[FLT]          \n\t"
        "mv             t2,     %[BIAS]         \n\t"
        "mv             t3,     %[DST]          \n\t"
        "mv             t4,     %[H_STRIDE]     \n\t"
        "mv             t5,     %[DST_H]        \n\t"
        "mv             t6,     %[DST_W]        \n\t"

        "addi           s3,     zero,   0       \n\t"
        "addi           s7,     zero,   4       \n\t"

        // load filter : v18-v26
        //      bias   : v31
        "vle.v          v18,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v19,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v20,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v21,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v22,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v23,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v24,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v25,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v26,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v31,    (t2)            \n\t"

        "0:                                     \n\t"
        "mv             s5,     t0              \n\t"
        "blt            t6,     s7,     2f      \n\t"
        "mv             s4,     t6              \n\t"
        // load src : v0-v17
        "1:                                     \n\t"
        "vle.v          v0,     (s5)            \n\t"
        "addi           s2,     s5,     16      \n\t"
        "vle.v          v1,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v2,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v3,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v4,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v5,     (s2)            \n\t"

        "add            s2,     s5,     t4      \n\t"
        "vle.v          v6,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v7,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v8,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v9,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v10,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v11,    (s2)            \n\t"

        "add            s2,     s5,     t4      \n\t"
        "add            s2,     s2,     t4      \n\t"
        "vle.v          v12,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v13,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v14,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v15,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v16,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v17,    (s2)            \n\t"
        // calculate
        "vmv.v.v        v27,    v31             \n\t"
        "vmv.v.v        v28,    v31             \n\t"
        "vmv.v.v        v29,    v31             \n\t"
        "vmv.v.v        v30,    v31             \n\t"

        "vfmacc.vv      v27,    v0,     v18     \n\t"
        "vfmacc.vv      v28,    v1,     v18     \n\t"
        "vfmacc.vv      v29,    v2,     v18     \n\t"
        "vfmacc.vv      v30,    v3,     v18     \n\t"

        "vfmacc.vv      v27,    v1,     v19     \n\t"
        "vfmacc.vv      v28,    v2,     v19     \n\t"
        "vfmacc.vv      v29,    v3,     v19     \n\t"
        "vfmacc.vv      v30,    v4,     v19     \n\t"

        "vfmacc.vv      v27,    v2,     v20     \n\t"
        "vfmacc.vv      v28,    v3,     v20     \n\t"
        "vfmacc.vv      v29,    v4,     v20     \n\t"
        "vfmacc.vv      v30,    v5,     v20     \n\t"

        "vfmacc.vv      v27,    v6,     v21     \n\t"
        "vfmacc.vv      v28,    v7,     v21     \n\t"
        "vfmacc.vv      v29,    v8,     v21     \n\t"
        "vfmacc.vv      v30,    v9,     v21     \n\t"

        "vfmacc.vv      v27,    v7,     v22     \n\t"
        "vfmacc.vv      v28,    v8,     v22     \n\t"
        "vfmacc.vv      v29,    v9,     v22     \n\t"
        "vfmacc.vv      v30,    v10,    v22     \n\t"

        "vfmacc.vv      v27,    v8,     v23     \n\t"
        "vfmacc.vv      v28,    v9,     v23     \n\t"
        "vfmacc.vv      v29,    v10,    v23     \n\t"
        "vfmacc.vv      v30,    v11,    v23     \n\t"

        "vfmacc.vv      v27,    v12,    v24     \n\t"
        "vfmacc.vv      v28,    v13,    v24     \n\t"
        "vfmacc.vv      v29,    v14,    v24     \n\t"
        "vfmacc.vv      v30,    v15,    v24     \n\t"

        "vfmacc.vv      v27,    v13,    v25     \n\t"
        "vfmacc.vv      v28,    v14,    v25     \n\t"
        "vfmacc.vv      v29,    v15,    v25     \n\t"
        "vfmacc.vv      v30,    v16,    v25     \n\t"

        "vfmacc.vv      v27,    v14,    v26     \n\t"
        "vfmacc.vv      v28,    v15,    v26     \n\t"
        "vfmacc.vv      v29,    v16,    v26     \n\t"
        "vfmacc.vv      v30,    v17,    v26     \n\t"
        // store dst    : v27-v30
        "vse.v          v27,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vse.v          v28,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vse.v          v29,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vse.v          v30,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"

        "addi           s4,     s4,     -4      \n\t"
        "addi           s5,     s5,     64      \n\t" // w_loop
        "bge            s4,     s7,     1b      \n\t"
        "beq            s4,     zero,   3f      \n\t"

        "2:                                     \n\t"
        "vle.v          v0,     (s5)            \n\t"
        "addi           s2,     s5,     16      \n\t"
        "vle.v          v1,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v2,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vle.v          v3,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vle.v          v4,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".endif                                 \n\t"

        "add            s2,     s5,     t4      \n\t"
        "vle.v          v6,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v7,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v8,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vle.v          v9,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vle.v          v10,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".endif                                 \n\t"

        "add            s2,     s5,     t4      \n\t"
        "add            s2,     s2,     t4      \n\t"
        "vle.v          v12,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v13,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        "vle.v          v14,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vle.v          v15,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vle.v          v16,    (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"
        ".endif                                 \n\t"
        // calculate
        "vmv.v.v        v27,    v31             \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vmv.v.v        v28,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vmv.v.v        v29,    v31             \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v0,     v18     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v1,     v18     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v2,     v18     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v1,     v19     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v2,     v19     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v3,     v19     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v2,     v20     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v3,     v20     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v4,     v20     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v6,     v21     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v7,     v21     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v8,     v21     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v7,     v22     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v8,     v22     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v9,     v22     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v8,     v23     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v9,     v23     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v10,    v23     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v12,    v24     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v13,    v24     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v14,    v24     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v13,    v25     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v14,    v25     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v15,    v25     \n\t"
        ".endif                                 \n\t"

        "vfmacc.vv      v27,    v14,    v26     \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vfmacc.vv      v28,    v15,    v26     \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vfmacc.vv      v29,    v16,    v26     \n\t"
        ".endif                                 \n\t"
        // store dst
        "vse.v          v27,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        ".if ATOM_W > 1                         \n\t"
        "vse.v          v28,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        ".endif                                 \n\t"
        ".if ATOM_W > 2                         \n\t"
        "vse.v          v29,    (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        ".endif                                 \n\t"

        "3:                                     \n\t"
        "addi           s3,     s3,     1       \n\t"
        "add            t0,     t0,     t4      \n\t" // h_loop
        "blt            s3,     t5,     0b      \n\t"
        :
        : [ATOM_W] "i"(atom_w), [SRC] "r"(src), [FLT] "r"(flt), [DST] "r"(dst), [BIAS] "r"(bias), [H_STRIDE] "r"(src_pad_w * 4 * 4), [DST_H] "r"(dst_h), [DST_W] "r"(dst_w)
        : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "s3", "s4", "s5", "s7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
