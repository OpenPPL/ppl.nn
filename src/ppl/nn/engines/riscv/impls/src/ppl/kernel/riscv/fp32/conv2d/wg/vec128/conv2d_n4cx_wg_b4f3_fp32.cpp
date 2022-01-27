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
#include "ppl/kernel/riscv/fp32/conv2d/wg/vec128/common/wg_pure.h"
#include "ppl/kernel/riscv/fp32/conv2d/wg/vec128/conv2d_n4cx_wg_b4f3_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/common/conv2d_shell_fp32.h"

namespace ppl { namespace kernel { namespace riscv {

void conv2d_n4cx_wg_b4f3_fp32_runtime_executor::adjust_tunning_param()
{
    auto dst_h = dst_shape_->GetDim(2);
    auto dst_w = dst_shape_->GetDim(3);

    tunning_param_.oh_blk = min(dst_h, tunning_param_.oh_blk);
    tunning_param_.ow_blk = min(dst_w, tunning_param_.ow_blk);

    tunning_param_.ic_blk = min(round_up(conv_param_->channels / conv_param_->group, 4), tunning_param_.ic_blk);
    tunning_param_.oc_blk = min(round_up(conv_param_->num_output / conv_param_->group, 4), tunning_param_.oc_blk);
}

ppl::common::RetCode conv2d_n4cx_wg_b4f3_fp32_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    adjust_tunning_param();
    LOG(DEBUG) << "n4cx wg b4f3: prepare";

    return ppl::common::RC_SUCCESS;
}

inline void wg_b4f3s1_src_trans_kernel(
    const float* src_pad,
    const float* trans_mat, // TODO: should be removed
    int64_t src_pad_h_stride,

    float* src_trans_d,
    int64_t src_trans_wg_tile_stride)
{
    float tmp[6][6][4];

    // perf method
    asm volatile(
        "mv             t0,     %[mat]          \n\t"
        "mv             t1,     %[src]          \n\t"
        "mv             t3,     %[tmp]          \n\t"
        "mv             t5,     %[src_offset]   \n\t"
        "addi           t2,     x0,     4       \n\t"
        "vsetvli        t6,     t2,     e32     \n\t"
        "vle.v          v0,     (t0)            \n\t"
        "vrgather.vi    v16,    v0,     0       \n\t" // 2
        "vrgather.vi    v17,    v0,     1       \n\t" // 4
        "vrgather.vi    v18,    v0,     2       \n\t" // 5
        "mv             t6,     t3              \n\t"
        "mv             s2,     x0              \n\t"
        "addi           t0,     x0,     6       \n\t"

        "1:                                     \n\t"
        "vle.v          v1,     (t1)            \n\t"
        "add            t2,     t1,     t5      \n\t"
        "vle.v          v2,     (t2)            \n\t"
        "add            t2,     t2,     t5      \n\t"
        "vle.v          v3,     (t2)            \n\t"
        "add            t2,     t2,     t5      \n\t"
        "vle.v          v4,     (t2)            \n\t"
        "add            t2,     t2,     t5      \n\t"
        "vle.v          v5,     (t2)            \n\t"
        "add            t2,     t2,     t5      \n\t"
        "vle.v          v6,     (t2)            \n\t"
        // tmp[0][j]
        "vfmul.vv       v20,    v3,     v18     \n\t"
        "vfmsac.vv      v20,    v1,     v17     \n\t"
        "vfadd.vv       v20,    v5,     v20     \n\t"
        "vse.v          v20,    (t6)            \n\t"
        // tmp[1][j] && tmp[2][j]
        "vfmul.vv       v30,    v3,     v17     \n\t"
        "vfsub.vv       v30,    v5,     v30     \n\t"
        "vfmul.vv       v31,    v2,     v17     \n\t"
        "vfsub.vv       v31,    v4,     v31     \n\t"
        "vfadd.vv       v21,    v30,    v31     \n\t"
        "addi           t4,     t6,     96      \n\t"
        "vse.v          v21,    (t4)            \n\t"
        "vfsub.vv       v22,    v30,    v31     \n\t"
        "addi           t4,     t4,     96      \n\t"
        "vse.v          v22,    (t4)            \n\t"
        // tmp[3][j] && tmp[4][j]
        "vfsub.vv       v30,    v2,     v4      \n\t"
        "vfmul.vv       v30,    v30,    v16     \n\t"
        "vfsub.vv       v31,    v5,     v3      \n\t"
        "vfsub.vv       v23,    v31,    v30     \n\t"
        "addi           t4,     t4,     96      \n\t"
        "vse.v          v23,    (t4)            \n\t"
        "vfadd.vv       v24,    v31,    v30     \n\t"
        "addi           t4,     t4,     96      \n\t"
        "vse.v          v24,    (t4)            \n\t"
        // tmp[5][j]
        "vfmul.vv       v25,    v4,     v18     \n\t"
        "vfmsac.vv      v25,    v2,     v17     \n\t"
        "vfadd.vv       v25,    v6,     v25     \n\t"
        "addi           t4,     t4,     96      \n\t"
        "vse.v          v25,    (t4)            \n\t"
        // loop acc
        "addi           t1,     t1,     16      \n\t"
        "addi           t6,     t6,     16      \n\t"
        "addi           s2,     s2,     1       \n\t"
        "bne            s2,     t0,     1b      \n\t"

        "mv             t1,     %[dst]          \n\t"
        "mv             t6,     %[dst_offset]   \n\t"
        "mv             t2,     x0              \n\t"
        "addi           t4,     x0,     6       \n\t"

        "2:                                     \n\t"
        "vle.v          v1,     (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vle.v          v2,     (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vle.v          v3,     (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vle.v          v4,     (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vle.v          v5,     (t3)            \n\t"
        "addi           t3,     t3,     16      \n\t"
        "vle.v          v6,     (t3)            \n\t"
        // dst[i][0]
        "vfmul.vv       v20,    v3,     v18     \n\t"
        "vfmsac.vv      v20,    v1,     v17     \n\t"
        "vfadd.vv       v20,    v5,     v20     \n\t"
        "vse.v          v20,    (t1)            \n\t"
        // dst[i][1] && dst[0][2]
        "vfmul.vv       v30,    v3,     v17     \n\t"
        "vfsub.vv       v30,    v5,     v30     \n\t"
        "vfmul.vv       v31,    v2,     v17     \n\t"
        "vfsub.vv       v31,    v4,     v31     \n\t"
        "vfadd.vv       v21,    v30,    v31     \n\t"
        "add            t1,     t1,     t6      \n\t"
        "vse.v          v21,    (t1)            \n\t"
        "vfsub.vv       v22,    v30,    v31     \n\t"
        "add            t1,     t1,     t6      \n\t"
        "vse.v          v22,    (t1)            \n\t"
        // dst[i][3] && dst[i][4]
        "vfsub.vv       v30,    v2,     v4      \n\t"
        "vfmul.vv       v30,    v30,    v16     \n\t"
        "vfsub.vv       v31,    v5,     v3      \n\t"
        "vfsub.vv       v23,    v31,    v30     \n\t"
        "add            t1,     t1,     t6      \n\t"
        "vse.v          v23,    (t1)            \n\t"
        "vfadd.vv       v24,    v31,    v30     \n\t"
        "add            t1,     t1,     t6      \n\t"
        "vse.v          v24,    (t1)            \n\t"
        // dst[i][5]
        "vfmul.vv       v25,    v4,     v18     \n\t"
        "vfmsac.vv      v25,    v2,     v17     \n\t"
        "vfadd.vv       v25,    v6,     v25     \n\t"
        "add            t1,     t1,     t6      \n\t"
        "vse.v          v25,    (t1)            \n\t"
        // loop acc
        "addi           t3,     t3,     16      \n\t"
        "add            t1,     t1,     t6      \n\t"
        "addi           t2,     t2,     1       \n\t"
        "bne            t2,     t4,     2b      \n\t"

        "addi           x0,     x0,     1       \n\t"
        :
        : [src] "r"(src_pad), [dst] "r"(src_trans_d), [mat] "r"(trans_mat), [tmp] "r"(tmp), [src_offset] "r"(src_pad_h_stride * 4), [dst_offset] "r"(src_trans_wg_tile_stride * 4)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v30", "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2");
}

inline void wg_b4f3s1_dst_trans_kernel(
    const float* dst_trans,
    const float* bias,
    const float* trans_mat, // TODO: should be removed
    int64_t dst_trans_wg_tile_stride,

    float* dst,
    int64_t dst_h_stride,
    int64_t dst_h_offset,
    int64_t dst_w_offset,
    int64_t dst_trans_h,
    int64_t dst_trans_w)
{
    // perf method
    asm volatile(
        // load trans_mat param
        "mv             t0,     %[mat]          \n\t"
        "flw            ft0,    (t0)            \n\t" // 2
        "flw            ft1,    4(t0)           \n\t" // 4
        "flw            ft2,    8(t0)           \n\t" // 8
        // load src
        "mv             t0,     %[src]          \n\t"
        "mv             t1,     %[src_offset1]  \n\t"
        "mv             t5,     %[src_offset0]  \n\t"
        "addi           t2,     x0,     4       \n\t"
        "vsetvli        t3,     t2,     e32     \n\t"
        // "mv             t2,     x0              \n\t"
        // "addi           t3,     x0,     6       \n\t"

        // "1:                                     \n\t"
        "vle.v          v0,     (t0)            \n\t"
        "add            t4,     t0,     t1      \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v5,     (t4)            \n\t"
        // calculate: common factor
        "vfadd.vv       v20,    v1,     v2      \n\t"
        "vfadd.vv       v6,     v3,     v4      \n\t"
        // calculate: tmp[0][j]
        "vfadd.vv       v8,     v20,    v6      \n\t"
        "vfadd.vv       v8,     v8,     v0      \n\t"
        // calculate: tmp[2][j]
        "vfmacc.vf      v20,    ft1,    v6      \n\t"
        // calculate: common factor
        "vfsub.vv       v14,    v1,     v2      \n\t"
        "vfsub.vv       v6,     v3,     v4      \n\t"
        // calculate: tmp[3][j]
        "vfadd.vv       v26,    v14,    v5      \n\t"
        "vfmacc.vf      v26,    ft2,    v6      \n\t"
        // calculate: tmp[1][j]
        "vfmacc.vf      v14,    ft0,    v6      \n\t"
        // loop acc
        "add            t0,     t0,     t5      \n\t"
        // "addi           t2,     t2,     1       \n\t"
        // "bne            t2,     t3,     1b      \n\t"
        "vle.v          v0,     (t0)            \n\t"
        "add            t4,     t0,     t1      \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v5,     (t4)            \n\t"
        "vfadd.vv       v21,    v1,     v2      \n\t"
        "vfadd.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v9,     v21,    v6      \n\t"
        "vfadd.vv       v9,     v9,     v0      \n\t"
        "vfmacc.vf      v21,    ft1,    v6      \n\t"
        "vfsub.vv       v15,    v1,     v2      \n\t"
        "vfsub.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v27,    v15,    v5      \n\t"
        "vfmacc.vf      v27,    ft2,    v6      \n\t"
        "vfmacc.vf      v15,    ft0,    v6      \n\t"
        "add            t0,     t0,     t5      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t4,     t0,     t1      \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v5,     (t4)            \n\t"
        "vfadd.vv       v22,    v1,     v2      \n\t"
        "vfadd.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v10,    v22,    v6      \n\t"
        "vfadd.vv       v10,    v10,    v0      \n\t"
        "vfmacc.vf      v22,    ft1,    v6      \n\t"
        "vfsub.vv       v16,    v1,     v2      \n\t"
        "vfsub.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v28,    v16,    v5      \n\t"
        "vfmacc.vf      v28,    ft2,    v6      \n\t"
        "vfmacc.vf      v16,    ft0,    v6      \n\t"
        "add            t0,     t0,     t5      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t4,     t0,     t1      \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v5,     (t4)            \n\t"
        "vfadd.vv       v23,    v1,     v2      \n\t"
        "vfadd.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v11,    v23,    v6      \n\t"
        "vfadd.vv       v11,    v11,    v0      \n\t"
        "vfmacc.vf      v23,    ft1,    v6      \n\t"
        "vfsub.vv       v17,    v1,     v2      \n\t"
        "vfsub.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v29,    v17,    v5      \n\t"
        "vfmacc.vf      v29,    ft2,    v6      \n\t"
        "vfmacc.vf      v17,    ft0,    v6      \n\t"
        "add            t0,     t0,     t5      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t4,     t0,     t1      \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v5,     (t4)            \n\t"
        "vfadd.vv       v24,    v1,     v2      \n\t"
        "vfadd.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v12,    v24,    v6      \n\t"
        "vfadd.vv       v12,    v12,    v0      \n\t"
        "vfmacc.vf      v24,    ft1,    v6      \n\t"
        "vfsub.vv       v18,    v1,     v2      \n\t"
        "vfsub.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v30,    v18,    v5      \n\t"
        "vfmacc.vf      v30,    ft2,    v6      \n\t"
        "vfmacc.vf      v18,    ft0,    v6      \n\t"
        "add            t0,     t0,     t5      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t4,     t0,     t1      \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     t1      \n\t"
        "vle.v          v5,     (t4)            \n\t"
        "vfadd.vv       v25,    v1,     v2      \n\t"
        "vfadd.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v13,    v25,    v6      \n\t"
        "vfadd.vv       v13,    v13,    v0      \n\t"
        "vfmacc.vf      v25,    ft1,    v6      \n\t"
        "vfsub.vv       v19,    v1,     v2      \n\t"
        "vfsub.vv       v6,     v3,     v4      \n\t"
        "vfadd.vv       v31,    v19,    v5      \n\t"
        "vfmacc.vf      v31,    ft2,    v6      \n\t"
        "vfmacc.vf      v19,    ft0,    v6      \n\t"

        "mv             t0,     %[dst]          \n\t"
        "mv             t1,     %[dst_offset]   \n\t"
        // "mv             t2,     x0              \n\t"
        "mv             t2,     %[h_offset]     \n\t"
        "mv             t3,     %[w_offset]     \n\t"
        "mv             t4,     %[dst_h]        \n\t"
        "mv             t5,     %[dst_w]        \n\t"

        // "2:                                     \n\t"
        // pad judge
        "bge            t2,     t4,     END     \n\t"
        // calculate: common factor
        "vle.v          v3,     (%[bias])       \n\t"
        "vfadd.vv       v0,     v9,     v10     \n\t"
        "vfadd.vv       v0,     v0,     v3      \n\t" // v0 = r01 + r02 + bias
        "vfadd.vv       v1,     v11,    v12     \n\t" // v1 = r03 + r04
        "vfsub.vv       v2,     v9,     v10     \n\t"
        "vfadd.vv       v2,     v2,     v3      \n\t" // v2 = r01 - r02 + bias
        "vfsub.vv       v3,     v11,    v12     \n\t" // v3 = r03 - r04
        // calculate: tmp[i][0] - tmp[i][3]
        "mv             t6,     t3              \n\t"
        "bge            t6,     t5,     L1      \n\t"
        "vfadd.vv       v4,     v0,     v1      \n\t"
        "vfadd.vv       v4,     v4,     v8      \n\t"
        "vse.v          v4,     (t0)            \n\t"
        "addi           s2,     t0,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L1      \n\t"
        "vfmul.vf       v5,     v3,     ft0     \n\t"
        "vfadd.vv       v5,     v5,     v2      \n\t"
        "vse.v          v5,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L1      \n\t"
        "vfmacc.vf      v0,     ft1,    v1      \n\t"
        "vse.v          v0,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L1      \n\t"
        "vfadd.vv       v7,     v2,     v13     \n\t"
        "vfmacc.vf      v7,     ft2,    v3      \n\t"
        "vse.v          v7,     (s2)            \n\t"
        // loop acc
        // "addi           t2,     t2,     1       \n\t"
        // "bne            t2,     t3,     2b      \n\t"
        "L1:                                    \n\t"
        "add            t0,     t0,     t1      \n\t"
        "addi           t2,     t2,     1       \n\t"
        "bge            t2,     t4,     END     \n\t"
        "vle.v          v3,     (%[bias])       \n\t"
        "vfadd.vv       v0,     v15,    v16     \n\t"
        "vfadd.vv       v0,     v0,     v3      \n\t"
        "vfadd.vv       v1,     v17,    v18     \n\t"
        "vfsub.vv       v2,     v15,    v16     \n\t"
        "vfadd.vv       v2,     v2,     v3      \n\t"
        "vfsub.vv       v3,     v17,    v18     \n\t"
        "mv             t6,     t3              \n\t"
        "bge            t6,     t5,     L2      \n\t"
        "vfadd.vv       v4,     v0,     v1      \n\t"
        "vfadd.vv       v4,     v4,     v14     \n\t"
        "vse.v          v4,     (t0)            \n\t"
        "addi           s2,     t0,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L2      \n\t"
        "vfmul.vf       v5,     v3,     ft0     \n\t"
        "vfadd.vv       v5,     v5,     v2      \n\t"
        "vse.v          v5,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L2      \n\t"
        "vfmacc.vf      v0,     ft1,    v1      \n\t"
        "vse.v          v0,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L2      \n\t"
        "vfadd.vv       v7,     v2,     v19     \n\t"
        "vfmacc.vf      v7,     ft2,    v3      \n\t"
        "vse.v          v7,     (s2)            \n\t"

        "L2:                                    \n\t"
        "add            t0,     t0,     t1      \n\t"
        "addi           t2,     t2,     1       \n\t"
        "bge            t2,     t4,     END     \n\t"
        "vle.v          v3,     (%[bias])       \n\t"
        "vfadd.vv       v0,     v21,    v22     \n\t"
        "vfadd.vv       v0,     v0,     v3      \n\t"
        "vfadd.vv       v1,     v23,    v24     \n\t"
        "vfsub.vv       v2,     v21,    v22     \n\t"
        "vfadd.vv       v2,     v2,     v3      \n\t"
        "vfsub.vv       v3,     v23,    v24     \n\t"
        "mv             t6,     t3              \n\t"
        "bge            t6,     t5,     L3      \n\t"
        "vfadd.vv       v4,     v0,     v1      \n\t"
        "vfadd.vv       v4,     v4,     v20     \n\t"
        "vse.v          v4,     (t0)            \n\t"
        "addi           s2,     t0,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L3      \n\t"
        "vfmul.vf       v5,     v3,     ft0      \n\t"
        "vfadd.vv       v5,     v5,     v2      \n\t"
        "vse.v          v5,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L3      \n\t"
        "vfmacc.vf      v0,     ft1,    v1      \n\t"
        "vse.v          v0,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L3      \n\t"
        "vfadd.vv       v7,     v2,     v25     \n\t"
        "vfmacc.vf      v7,     ft2,    v3      \n\t"
        "vse.v          v7,     (s2)            \n\t"

        "L3:                                    \n\t"
        "add            t0,     t0,     t1      \n\t"
        "addi           t2,     t2,     1       \n\t"
        "bge            t2,     t4,     END     \n\t"
        "vle.v          v3,     (%[bias])       \n\t"
        "vfadd.vv       v0,     v27,    v28     \n\t"
        "vfadd.vv       v0,     v0,     v3      \n\t"
        "vfadd.vv       v1,     v29,    v30     \n\t"
        "vfsub.vv       v2,     v27,    v28     \n\t"
        "vfadd.vv       v2,     v2,     v3      \n\t"
        "vfsub.vv       v3,     v29,    v30     \n\t"
        "mv             t6,     t3              \n\t"
        "bge            t6,     t5,     END     \n\t"
        "vfadd.vv       v4,     v0,     v1      \n\t"
        "vfadd.vv       v4,     v4,     v26     \n\t"
        "vse.v          v4,     (t0)            \n\t"
        "addi           s2,     t0,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     END     \n\t"
        "vfmul.vf       v5,     v3,     ft0     \n\t"
        "vfadd.vv       v5,     v5,     v2      \n\t"
        "vse.v          v5,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     END     \n\t"
        "vfmacc.vf      v0,     ft1,    v1      \n\t"
        "vse.v          v0,     (s2)            \n\t"
        "addi           s2,     s2,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     END     \n\t"
        "vfadd.vv       v7,     v2,     v31     \n\t"
        "vfmacc.vf      v7,     ft2,    v3      \n\t"
        "vse.v          v7,     (s2)            \n\t"

        "END:                                   \n\t"
        "addi           x0,     x0,     1       \n\t"
        :
        : [src] "r"(dst_trans), [dst] "r"(dst), [mat] "r"(trans_mat), [bias] "r"(bias), [src_offset0] "r"(dst_trans_wg_tile_stride * 4), [dst_offset] "r"(dst_h_stride * 4), [src_offset1] "r"(dst_trans_wg_tile_stride * 24), [h_offset] "r"(dst_h_offset), [w_offset] "r"(dst_w_offset), [dst_h] "r"(dst_trans_h), [dst_w] "r"(dst_trans_w)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "ft0", "ft1", "ft2");
}

ppl::common::RetCode conv2d_n4cx_wg_b4f3_fp32_runtime_executor::execute()
{
    const conv2d_common_param& cp = *conv_param_;

    LOG(DEBUG) << "n4cx wg b4f3: execute";
    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const float trans_mat_src_[8] = {2.0f, 4.0f, 5.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float trans_mat_dst_[3] = {2.0f, 4.0f, 8.0f};

    conv2d_shell_fp32<
        conv2d_n4cx_wg_bxfxs1_fp32_vec128_extra_param,
        4,
        conv2d_get_real_filter_size<4, 3>,
        conv2d_conv_wg_bxfxs1_riscv_per_group_fp32<4, 3, wg_b4f3s1_src_trans_kernel, wg_b4f3s1_dst_trans_kernel>>(
        src_,
        cvt_filter_,
        cvt_bias_,
        (float*)temp_buffer_,
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
