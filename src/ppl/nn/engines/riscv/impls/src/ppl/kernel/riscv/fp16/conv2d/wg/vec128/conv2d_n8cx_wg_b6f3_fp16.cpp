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

#include "ppl/common/log.h"
#include "ppl/kernel/riscv/common/math.h"
#include "ppl/kernel/riscv/fp16/conv2d/wg/vec128/common/wg_pure.h"
#include "ppl/kernel/riscv/fp16/conv2d/wg/vec128/conv2d_n8cx_wg_b6f3_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/conv_shell.h"

namespace ppl { namespace kernel { namespace riscv {

void conv2d_n8cx_wg_b6f3_fp16_runtime_executor::adjust_tunning_param()
{
    auto dst_h = dst_shape_->GetDim(2);
    auto dst_w = dst_shape_->GetDim(3);

    tunning_param_.oh_blk = min(dst_h, tunning_param_.oh_blk);
    tunning_param_.ow_blk = min(dst_w, tunning_param_.ow_blk);

    tunning_param_.ic_blk = min(round_up(conv_param_->channels / conv_param_->group, 8), tunning_param_.ic_blk);
    tunning_param_.oc_blk = min(round_up(conv_param_->num_output / conv_param_->group, 8), tunning_param_.oc_blk);
}

ppl::common::RetCode conv2d_n8cx_wg_b6f3_fp16_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    adjust_tunning_param();
    LOG(DEBUG) << "n8cx wg b6f3: prepare";

    return ppl::common::RC_SUCCESS;
}

inline void wg_b6f3s1_src_trans_kernel(
    const __fp16* src_pad,
    const __fp16* trans_mat, // TODO: should be removed
    int64_t src_pad_h_stride,

    __fp16* src_trans_d,
    int64_t src_trans_wg_tile_stride)
{
    __fp16 tmp[8][8][8];

    // perf method
    asm volatile(
        "addi           t0,     x0,     8               \n\t"
        "vsetvli        t1,     t0,     e16             \n\t"
        "mv             s2,     %[src]                  \n\t"
        "mv             s3,     %[tmp]                  \n\t"
        "mv             s4,     %[dst]                  \n\t"
        // load trans_mat param
        "flw            f0,     (%[mat])                \n\t" // 5.25
        "flw            f1,     4(%[mat])               \n\t" // 4.25
        "flw            f2,     8(%[mat])               \n\t" // 0.5
        "flw            f3,     12(%[mat])              \n\t" // 0.25
        "flw            f4,     16(%[mat])              \n\t" // 2.5
        "flw            f5,     20(%[mat])              \n\t" // 1.25
        "flw            f6,     24(%[mat])              \n\t" // 2
        "flw            f7,     28(%[mat])              \n\t" // 4

        // Step: 0
        "mv             t4,     s3                      \n\t"
        "mv             t1,     x0                      \n\t"
        "addi           t2,     x0,         8           \n\t"

        "1:                                             \n\t"
        // load src
        "vle.v          v0,     (s2)                    \n\t"
        "add            t0,     s2,         %[src_ofst] \n\t"
        "vle.v          v1,     (t0)                    \n\t"
        "add            t0,     t0,         %[src_ofst] \n\t"
        "vle.v          v2,     (t0)                    \n\t"
        "add            t0,     t0,         %[src_ofst] \n\t"
        "vle.v          v3,     (t0)                    \n\t"
        "add            t0,     t0,         %[src_ofst] \n\t"
        "vle.v          v4,     (t0)                    \n\t"
        "add            t0,     t0,         %[src_ofst] \n\t"
        "vle.v          v5,     (t0)                    \n\t"
        "add            t0,     t0,         %[src_ofst] \n\t"
        "vle.v          v6,     (t0)                    \n\t"
        "add            t0,     t0,         %[src_ofst] \n\t"
        "vle.v          v7,     (t0)                    \n\t"
        // tmp[0][j]
        "vfsub.vv       v8,     v0,         v6          \n\t"
        "vfsub.vv       v24,    v4,         v2          \n\t"
        "vfmacc.vf      v8,     f0,         v24         \n\t"
        "vse.v          v8,     (t4)                    \n\t"
        // tmp[1][j] && tmp[2][j]
        "addi           t3,     t4,         128         \n\t"
        "vfadd.vv       v24,    v2,         v6          \n\t"
        "vfmul.vf       v25,    v4,         f1          \n\t"
        "vfsub.vv       v30,    v24,        v25         \n\t"
        "vfadd.vv       v24,    v1,         v5          \n\t"
        "vfmul.vf       v25,    v3,         f1          \n\t"
        "vfsub.vv       v31,    v24,        v25         \n\t"
        "vfadd.vv       v9,     v30,        v31         \n\t"
        "vse.v          v9,     (t3)                    \n\t"
        "vfsub.vv       v10,    v30,        v31         \n\t"
        "addi           t3,     t3,         128         \n\t"
        "vse.v          v10,    (t3)                    \n\t"
        // tmp[3][j] && tmp[4][j]
        "vfmul.vf       v24,    v4,         f5          \n\t"
        "vmv.v.v        v25,    v24                     \n\t"
        "vfmsac.vf      v24,    f3,         v2          \n\t"
        "vfadd.vv       v30,    v24,        v6          \n\t"
        "vfmul.vf       v31,    v3,         f4          \n\t"
        "vmv.v.v        v26,    v31                     \n\t"
        "vfmsac.vf      v31,    f2,         v1          \n\t"
        "vfmacc.vf      v31,    f6,         v5          \n\t"
        "addi           t3,     t3,         128         \n\t"
        "vfadd.vv       v11,    v30,        v31         \n\t"
        "vse.v          v11,    (t3)                    \n\t"
        "addi           t3,     t3,         128         \n\t"
        "vfsub.vv       v12,    v30,        v31         \n\t"
        "vse.v          v12,    (t3)                    \n\t"
        // tmp[5][j] && tmp[6][j]
        "vfsub.vv       v24,    v2,         v25         \n\t"
        "vfmul.vf       v30,    v24,        f7          \n\t"
        "vfadd.vv       v30,    v30,        v6          \n\t"
        "vfmul.vf       v31,    v1,         f6          \n\t"
        "vfmacc.vf      v31,    f2,         v5          \n\t"
        "vfsub.vv       v31,    v31,        v26         \n\t"
        "addi           t3,     t3,         128         \n\t"
        "vfadd.vv       v13,    v30,        v31         \n\t"
        "vse.v          v13,    (t3)                    \n\t"
        "addi           t3,     t3,         128         \n\t"
        "vfsub.vv       v14,    v30,        v31         \n\t"
        "vse.v          v14,    (t3)                    \n\t"
        // tmp[7][j]
        "vfsub.vv       v15,    v7,         v1          \n\t"
        "vfsub.vv       v24,    v3,         v5          \n\t"
        "addi           t3,     t3,         128         \n\t"
        "vfmacc.vf      v15,    f0,         v24         \n\t"
        "vse.v          v15,    (t3)                    \n\t"
        // loop acc
        "addi           s2,     s2,         16          \n\t"
        "addi           t4,     t4,         16          \n\t"
        "addi           t1,     t1,         1           \n\t"
        "bne            t1,     t2,         1b          \n\t"

        // Step: 1
        "mv             t1,     x0                      \n\t"

        "2:                                             \n\t"
        "vle.v          v0,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v1,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v2,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v3,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v4,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v5,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v6,     (s3)                    \n\t"
        "addi           s3,     s3,         16          \n\t"
        "vle.v          v7,     (s3)                    \n\t"
        // dst[i][0]
        "vfsub.vv       v8,     v0,         v6          \n\t"
        "vfsub.vv       v24,    v4,         v2          \n\t"
        "vfmacc.vf      v8,     f0,         v24         \n\t"
        "vse.v          v8,     (s4)                    \n\t"
        // dst[i][1] && dst[i][2]
        "vfadd.vv       v24,    v2,         v6          \n\t"
        "vfmul.vf       v25,    v4,         f1          \n\t"
        "vfsub.vv       v30,    v24,        v25         \n\t"
        "vfadd.vv       v24,    v1,         v5          \n\t"
        "vfmul.vf       v25,    v3,         f1          \n\t"
        "vfsub.vv       v31,    v24,        v25         \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfadd.vv       v9,     v30,        v31         \n\t"
        "vse.v          v9,     (s4)                    \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfsub.vv       v10,    v30,        v31         \n\t"
        "vse.v          v10,    (s4)                    \n\t"
        // dst[i][3] && dst[i][4]
        "vfmul.vf       v24,    v4,         f5          \n\t"
        "vmv.v.v        v25,    v24                     \n\t"
        "vfmsac.vf      v24,    f3,         v2          \n\t"
        "vfadd.vv       v30,    v24,        v6          \n\t"
        "vfmul.vf       v31,    v3,         f4          \n\t"
        "vmv.v.v        v26,    v31                     \n\t"
        "vfmsac.vf      v31,    f2,         v1          \n\t"
        "vfmacc.vf      v31,    f6,         v5          \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfadd.vv       v11,    v30,        v31         \n\t"
        "vse.v          v11,    (s4)                    \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfsub.vv       v12,    v30,        v31         \n\t"
        "vse.v          v12,    (s4)                    \n\t"
        // dst[i][5] && dst[i][6]
        "vfsub.vv       v24,    v2,         v25         \n\t"
        "vfmul.vf       v30,    v24,        f7          \n\t"
        "vfadd.vv       v30,    v30,        v6          \n\t"
        "vfmul.vf       v31,    v1,         f6          \n\t"
        "vfmacc.vf      v31,    f2,         v5          \n\t"
        "vfsub.vv       v31,    v31,        v26         \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfadd.vv       v13,    v30,        v31         \n\t"
        "vse.v          v13,    (s4)                    \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfsub.vv       v14,    v30,        v31         \n\t"
        "vse.v          v14,    (s4)                    \n\t"
        // dst[i][7]
        "vfsub.vv       v15,    v7,         v1          \n\t"
        "vfsub.vv       v24,    v3,         v5          \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "vfmacc.vf      v15,    f0,         v24         \n\t"
        "vse.v          v15,    (s4)                    \n\t"
        // loop acc
        "addi           s3,     s3,         16          \n\t"
        "add            s4,     s4,         %[dst_ofst] \n\t"
        "addi           t1,     t1,         1           \n\t"
        "bne            t1,     t2,         2b          \n\t"

        "addi           x0,     x0,         1           \n\t"
        :
        : [src] "r"(src_pad), [dst] "r"(src_trans_d), [mat] "r"(trans_mat), [tmp] "r"(tmp), [src_ofst] "r"(src_pad_h_stride * 2), [dst_ofst] "r"(src_trans_wg_tile_stride * 2)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v30", "v31", "t0", "t1", "t2", "t3", "t4", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "s2", "s3", "s4");
}

inline void wg_b6f3s1_dst_trans_kernel(
    const __fp16* dst_trans,
    const __fp16* bias,
    const __fp16* trans_mat, // TODO: should be removed
    int64_t dst_trans_wg_tile_stride,

    __fp16* dst,
    int64_t dst_h_stride,
    int64_t dst_h_offset,
    int64_t dst_w_offset,
    int64_t dst_trans_h,
    int64_t dst_trans_w)
{
    __fp16 tmp[6][8][8];

    // perf method
    asm volatile(
        "addi           t0,         x0,     8               \n\t"
        "vsetvli        t1,         t0,     e16             \n\t"
        "mv             s2,         %[src]                  \n\t"
        "mv             s3,         %[tmp]                  \n\t"
        "mv             s4,         %[dst]                  \n\t"
        "mv             s5,         %[h_ofst]               \n\t"
        // load trans_mat param
        "flw            f0,         (%[mat])                \n\t" // 32
        "flw            f1,         4(%[mat])               \n\t" // 16
        "flw            f2,         8(%[mat])               \n\t" // 8
        "flw            f3,         12(%[mat])              \n\t" // 4
        "flw            f4,         16(%[mat])              \n\t" // 2

        "mv             t0,         s3                      \n\t"
        "mv             t1,         x0                      \n\t"
        "addi           t2,         x0,         8           \n\t"

        "1:                                                 \n\t"
        // load src
        "vle.v          v0,         (s2)                    \n\t"
        "add            t3,         s2,         %[src_ofst0]\n\t"
        "vle.v          v1,         (t3)                    \n\t"
        "add            t3,         t3,         %[src_ofst0]\n\t"
        "vle.v          v2,         (t3)                    \n\t"
        "add            t3,         t3,         %[src_ofst0]\n\t"
        "vle.v          v3,         (t3)                    \n\t"
        "add            t3,         t3,         %[src_ofst0]\n\t"
        "vle.v          v4,         (t3)                    \n\t"
        "add            t3,         t3,         %[src_ofst0]\n\t"
        "vle.v          v5,         (t3)                    \n\t"
        "add            t3,         t3,         %[src_ofst0]\n\t"
        "vle.v          v6,         (t3)                    \n\t"
        "add            t3,         t3,         %[src_ofst0]\n\t"
        "vle.v          v7,         (t3)                    \n\t"
        // common factor
        "vfadd.vv       v16,        v1,         v2          \n\t"
        "vfsub.vv       v17,        v1,         v2          \n\t"
        "vfadd.vv       v18,        v3,         v4          \n\t"
        "vfsub.vv       v19,        v3,         v4          \n\t"
        "vfadd.vv       v20,        v5,         v6          \n\t"
        "vfsub.vv       v21,        v5,         v6          \n\t"
        // tmp[0][j]
        "vfadd.vv       v8,         v0,         v16         \n\t"
        "vfadd.vv       v8,         v8,         v18         \n\t"
        "vfmacc.vf      v8,         f0,         v20         \n\t"
        "vse.v          v8,         (t0)                    \n\t"
        // tmp[1][j]
        "vfmul.vf       v9,         v19,        f4          \n\t"
        "vfmacc.vf      v9,         f1,         v21         \n\t"
        "vfadd.vv       v9,         v9,         v17         \n\t"
        "addi           t4,         t0,         128         \n\t"
        "vse.v          v9,         (t4)                    \n\t"
        // tmp[2][j]
        "vfmul.vf       v10,        v18,        f3          \n\t"
        "vfmacc.vf      v10,        f2,         v20         \n\t"
        "vfadd.vv       v10,        v10,        v16         \n\t"
        "addi           t4,         t4,         128         \n\t"
        "vse.v          v10,        (t4)                    \n\t"
        // tmp[3][j]
        "vfmul.vf       v11,        v19,        f2          \n\t"
        "vfmacc.vf      v11,        f3,         v21         \n\t"
        "vfadd.vv       v11,        v11,        v17         \n\t"
        "addi           t4,         t4,         128         \n\t"
        "vse.v          v11,        (t4)                    \n\t"
        // tmp[4][j]
        "vfmul.vf       v12,        v18,        f1          \n\t"
        "vfmacc.vf      v12,        f4,         v20         \n\t"
        "vfadd.vv       v12,        v12,        v16         \n\t"
        "addi           t4,         t4,         128         \n\t"
        "vse.v          v12,        (t4)                    \n\t"
        // tmp[5][j]
        "vfadd.vv       v13,        v7,         v17         \n\t"
        "vfmacc.vf      v13,        f0,         v19         \n\t"
        "vfadd.vv       v13,        v13,        v21         \n\t"
        "addi           t4,         t4,         128         \n\t"
        "vse.v          v13,        (t4)                    \n\t"
        // loop acc
        "add            s2,         s2,         %[src_ofst1]\n\t"
        "addi           t0,         t0,         16          \n\t"
        "addi           t1,         t1,         1           \n\t"
        "bne            t1,         t2,         1b          \n\t"

        "mv             t1,         x0                      \n\t"
        "addi           t2,         x0,         6           \n\t"
        "vle.v          v24,        (%[bias])               \n\t"

        "2:                                                 \n\t"
        "bge            s5,         %[dst_h],   END         \n\t"
        // load tmp
        "vle.v          v0,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v1,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v2,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v3,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v4,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v5,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v6,         (s3)                    \n\t"
        "addi           s3,         s3,         16          \n\t"
        "vle.v          v7,         (s3)                    \n\t"
        // common factor
        "vfadd.vv       v16,        v1,         v2          \n\t"
        "vfadd.vv       v16,        v16,        v24         \n\t" // v16 = r1 + r2 + bias
        "vfsub.vv       v17,        v1,         v2          \n\t"
        "vfadd.vv       v17,        v17,        v24         \n\t" // v17 = r1 - r2 + bias
        "vfadd.vv       v18,        v3,         v4          \n\t"
        "vfsub.vv       v19,        v3,         v4          \n\t"
        "vfadd.vv       v20,        v5,         v6          \n\t"
        "vfsub.vv       v21,        v5,         v6          \n\t"
        "mv             t3,         %[w_ofst]               \n\t"
        // tmp[i][0]
        "bge            t3,         %[dst_w],   3f          \n\t"
        "vfadd.vv       v8,         v0,         v16         \n\t"
        "vfadd.vv       v8,         v8,         v18         \n\t"
        "vfmacc.vf      v8,         f0,         v20         \n\t"
        "vse.v          v8,         (s4)                    \n\t"
        "addi           t0,         s4,         16          \n\t"
        // tmp[i][1]
        "addi           t3,         t3,         1           \n\t"
        "bge            t3,         %[dst_w],   3f          \n\t"
        "vfmul.vf       v9,         v19,        f4          \n\t"
        "vfmacc.vf      v9,         f1,         v21         \n\t"
        "vfadd.vv       v9,         v9,         v17         \n\t"
        "vse.v          v9,         (t0)                    \n\t"
        "addi           t0,         t0,         16          \n\t"
        // tmp[i][2]
        "addi           t3,         t3,         1           \n\t"
        "bge            t3,         %[dst_w],   3f          \n\t"
        "vfmul.vf       v10,        v18,        f3          \n\t"
        "vfmacc.vf      v10,        f2,         v20         \n\t"
        "vfadd.vv       v10,        v10,        v16         \n\t"
        "vse.v          v10,        (t0)                    \n\t"
        "addi           t0,         t0,         16          \n\t"
        // tmp[i][3]
        "addi           t3,         t3,         1           \n\t"
        "bge            t3,         %[dst_w],   3f          \n\t"
        "vfmul.vf       v11,        v19,        f2          \n\t"
        "vfmacc.vf      v11,        f3,         v21         \n\t"
        "vfadd.vv       v11,        v11,        v17         \n\t"
        "vse.v          v11,        (t0)                    \n\t"
        "addi           t0,         t0,         16          \n\t"
        // tmp[i][4]
        "addi           t3,         t3,         1           \n\t"
        "bge            t3,         %[dst_w],   3f          \n\t"
        "vfmul.vf       v12,        v18,        f1          \n\t"
        "vfmacc.vf      v12,        f4,         v20         \n\t"
        "vfadd.vv       v12,        v12,        v16         \n\t"
        "vse.v          v12,        (t0)                    \n\t"
        "addi           t0,         t0,         16          \n\t"
        // tmp[i][5]
        "addi           t3,         t3,         1           \n\t"
        "bge            t3,         %[dst_w],   3f          \n\t"
        "vfadd.vv       v13,        v7,         v17         \n\t"
        "vfmacc.vf      v13,        f0,         v19         \n\t"
        "vfadd.vv       v13,        v13,        v21         \n\t"
        "vse.v          v13,        (t0)                    \n\t"
        // loop acc
        "3:                                                 \n\t"
        "addi           s5,         s5,         1           \n\t"
        "add            s4,         s4,         %[dst_ofst] \n\t"
        "addi           s3,         s3,         16          \n\t"
        "addi           t1,         t1,         1           \n\t"
        "bne            t1,         t2,         2b          \n\t"

        "END:                                               \n\t"
        "addi           x0,         x0,         1           \n\t"
        :
        : [src] "r"(dst_trans), [dst] "r"(dst), [mat] "r"(trans_mat), [tmp] "r"(tmp), [bias] "r"(bias), [src_ofst0] "r"(dst_trans_wg_tile_stride * 16), [dst_ofst] "r"(dst_h_stride * 2), [src_ofst1] "r"(dst_trans_wg_tile_stride * 2), [h_ofst] "r"(dst_h_offset), [w_ofst] "r"(dst_w_offset), [dst_h] "r"(dst_trans_h), [dst_w] "r"(dst_trans_w)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v24", "t0", "t1", "t2", "t3", "t4", "f0", "f1", "f2", "f3", "f4", "s2", "s3", "s4", "s5");
}

ppl::common::RetCode conv2d_n8cx_wg_b6f3_fp16_runtime_executor::execute()
{
    const conv2d_common_param& cp = *conv_param_;

    LOG(DEBUG) << "n8cx wg b6f3: execute";
    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const __fp16 trans_mat_src_[16]   = {5.25f, 0.0f, 4.25f, 0.0f, 0.5f, 0.0f, 0.25f, 0.0f, 2.5f, 0.0f, 1.25f, 0.0f, 2.0f, 0.0f, 4.0f, 0.0f};
    ((__uint16_t*)trans_mat_src_)[1]  = 0xffff;
    ((__uint16_t*)trans_mat_src_)[3]  = 0xffff;
    ((__uint16_t*)trans_mat_src_)[5]  = 0xffff;
    ((__uint16_t*)trans_mat_src_)[7]  = 0xffff;
    ((__uint16_t*)trans_mat_src_)[9]  = 0xffff;
    ((__uint16_t*)trans_mat_src_)[11] = 0xffff;
    ((__uint16_t*)trans_mat_src_)[13] = 0xffff;
    ((__uint16_t*)trans_mat_src_)[15] = 0xffff;

    const __fp16 trans_mat_dst_[10]  = {32.0f, 0.0f, 16.0f, 0.0f, 8.0f, 0.0f, 4.0f, 0.0f, 2.0f, 0.0f};
    ((__uint16_t*)trans_mat_dst_)[1] = 0xffff;
    ((__uint16_t*)trans_mat_dst_)[3] = 0xffff;
    ((__uint16_t*)trans_mat_dst_)[5] = 0xffff;
    ((__uint16_t*)trans_mat_dst_)[7] = 0xffff;
    ((__uint16_t*)trans_mat_dst_)[9] = 0xffff;

    conv_shell_riscv_fp16<
        conv2d_n8cx_wg_bxfxs1_fp16_vec128_extra_param,
        8,
        get_real_filter_size<6, 3>,
        conv_wg_bxfxs1_riscv_per_group_fp16<6, 3, wg_b6f3s1_src_trans_kernel, wg_b6f3s1_dst_trans_kernel>>(
        src_,
        cvt_filter_,
        cvt_bias_,
        (__fp16*)temp_buffer_,
        dst_,

        src_shape_->GetDim(2),
        src_shape_->GetDim(3),
        conv_param_->pad_h,
        conv_param_->pad_w,
        conv_param_->kernel_h,
        conv_param_->kernel_w,
        conv_param_->stride_h,
        conv_param_->stride_w,
        conv_param_->dilation_h,
        conv_param_->dilation_w,
        conv_param_->channels,
        conv_param_->num_output,
        conv_param_->group,
        src_shape_->GetDim(0),

        {tunning_param_.oc_blk,
         tunning_param_.ic_blk,
         tunning_param_.oh_blk,
         tunning_param_.ow_blk,

         trans_mat_src_,
         trans_mat_dst_});

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv
