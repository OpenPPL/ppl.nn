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
#include "ppl/kernel/riscv/fp16/conv2d/wg/vec128/conv2d_n8cx_wg_b2f3_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/conv_shell.h"

namespace ppl { namespace kernel { namespace riscv {

void conv2d_n8cx_wg_b2f3_fp16_runtime_executor::adjust_tunning_param()
{
    auto dst_h = dst_shape_->GetDim(2);
    auto dst_w = dst_shape_->GetDim(3);

    tunning_param_.oh_blk = min(dst_h, tunning_param_.oh_blk);
    tunning_param_.ow_blk = min(dst_w, tunning_param_.ow_blk);

    tunning_param_.ic_blk = min(round_up(conv_param_->channels / conv_param_->group, 8), tunning_param_.ic_blk);
    tunning_param_.oc_blk = min(round_up(conv_param_->num_output / conv_param_->group, 8), tunning_param_.oc_blk);
}

ppl::common::RetCode conv2d_n8cx_wg_b2f3_fp16_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    adjust_tunning_param();
    LOG(DEBUG) << "n8cx wg b2f3: prepare";

    return ppl::common::RC_SUCCESS;
}

inline void wg_b2f3s1_src_trans_kernel(
    const __fp16* src_pad,
    const __fp16* trans_mat, // TODO: should be removed
    int64_t src_pad_h_stride,

    __fp16* src_trans_d,
    int64_t src_trans_wg_tile_stride)
{
    // perf method
    asm volatile(
        "mv             t0,     %[src]          \n\t"
        "mv             t1,     %[src_offset]   \n\t"
        "addi           t2,     x0,     8       \n\t"
        "vsetvli        t6,     t2,     e16     \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t2,     t0,     t1      \n\t"
        "vle.v          v1,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v2,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v3,     (t2)            \n\t"
        "vfsub.vv       v16,    v0,     v2      \n\t"
        "vfadd.vv       v20,    v1,     v2      \n\t"
        "vfsub.vv       v24,    v2,     v1      \n\t"
        "vfsub.vv       v28,    v3,     v1      \n\t"

        "addi           t0,     t0,     16      \n\t"
        "vle.v          v0,     (t0)            \n\t"
        "add            t2,     t0,     t1      \n\t"
        "vle.v          v1,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v2,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v3,     (t2)            \n\t"
        "vfsub.vv       v17,    v0,     v2      \n\t"
        "vfadd.vv       v21,    v1,     v2      \n\t"
        "vfsub.vv       v25,    v2,     v1      \n\t"
        "vfsub.vv       v29,    v3,     v1      \n\t"

        "addi           t0,     t0,     16      \n\t"
        "vle.v          v0,     (t0)            \n\t"
        "add            t2,     t0,     t1      \n\t"
        "vle.v          v1,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v2,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v3,     (t2)            \n\t"
        "vfsub.vv       v18,    v0,     v2      \n\t"
        "vfadd.vv       v22,    v1,     v2      \n\t"
        "vfsub.vv       v26,    v2,     v1      \n\t"
        "vfsub.vv       v30,    v3,     v1      \n\t"

        "addi           t0,     t0,     16      \n\t"
        "vle.v          v0,     (t0)            \n\t"
        "add            t2,     t0,     t1      \n\t"
        "vle.v          v1,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v2,     (t2)            \n\t"
        "add            t2,     t2,     t1      \n\t"
        "vle.v          v3,     (t2)            \n\t"
        "vfsub.vv       v19,    v0,     v2      \n\t"
        "vfadd.vv       v23,    v1,     v2      \n\t"
        "vfsub.vv       v27,    v2,     v1      \n\t"
        "vfsub.vv       v31,    v3,     v1      \n\t"

        "mv             t0,     %[dst]          \n\t"
        "mv             t1,     %[dst_offset]   \n\t"

        "vfsub.vv       v0,     v16,    v18     \n\t"
        "vse.v          v0,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfadd.vv       v1,     v17,    v18     \n\t"
        "vse.v          v1,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v2,     v18,    v17     \n\t"
        "vse.v          v2,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v3,     v19,    v17     \n\t"
        "vse.v          v3,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"

        "vfsub.vv       v0,     v20,    v22     \n\t"
        "vse.v          v0,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfadd.vv       v1,     v21,    v22     \n\t"
        "vse.v          v1,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v2,     v22,    v21     \n\t"
        "vse.v          v2,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v3,     v23,    v21     \n\t"
        "vse.v          v3,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"

        "vfsub.vv       v0,     v24,    v26     \n\t"
        "vse.v          v0,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfadd.vv       v1,     v25,    v26     \n\t"
        "vse.v          v1,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v2,     v26,    v25     \n\t"
        "vse.v          v2,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v3,     v27,    v25     \n\t"
        "vse.v          v3,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"

        "vfsub.vv       v0,     v28,    v30     \n\t"
        "vse.v          v0,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfadd.vv       v1,     v29,    v30     \n\t"
        "vse.v          v1,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v2,     v30,    v29     \n\t"
        "vse.v          v2,     (t0)            \n\t"
        "add            t0,     t0,     t1      \n\t"
        "vfsub.vv       v3,     v31,    v29     \n\t"
        "vse.v          v3,     (t0)            \n\t"
        :
        : [src] "r"(src_pad), [dst] "r"(src_trans_d), [src_offset] "r"(src_pad_h_stride * 2), [dst_offset] "r"(src_trans_wg_tile_stride * 2)
        : "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "t0", "t1", "t2", "t6");
}

inline void wg_b2f3s1_dst_trans_kernel(
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
    // perf method
    asm volatile(
        "addi           t0,     x0,     8       \n\t"
        "vsetvli        t1,     t0,     e16     \n\t"

        "mv             t0,     %[src]          \n\t"
        "mv             t1,     %[src_offset0]  \n\t"
        "mv             t2,     %[src_offset1]  \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t3,     t0,     t1      \n\t"
        "vle.v          v1,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v2,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v3,     (t3)            \n\t"
        "vfadd.vv       v16,    v0,     v1      \n\t"
        "vfadd.vv       v16,    v16,    v2      \n\t"
        "vfsub.vv       v20,    v1,     v2      \n\t"
        "vfadd.vv       v20,    v20,    v3      \n\t"
        "add            t0,     t0,     t2      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t3,     t0,     t1      \n\t"
        "vle.v          v1,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v2,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v3,     (t3)            \n\t"
        "vfadd.vv       v17,    v0,     v1      \n\t"
        "vfadd.vv       v17,    v17,    v2      \n\t"
        "vfsub.vv       v21,    v1,     v2      \n\t"
        "vfadd.vv       v21,    v21,    v3      \n\t"
        "add            t0,     t0,     t2      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t3,     t0,     t1      \n\t"
        "vle.v          v1,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v2,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v3,     (t3)            \n\t"
        "vfadd.vv       v18,    v0,     v1      \n\t"
        "vfadd.vv       v18,    v18,    v2      \n\t"
        "vfsub.vv       v22,    v1,     v2      \n\t"
        "vfadd.vv       v22,    v22,    v3      \n\t"
        "add            t0,     t0,     t2      \n\t"

        "vle.v          v0,     (t0)            \n\t"
        "add            t3,     t0,     t1      \n\t"
        "vle.v          v1,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v2,     (t3)            \n\t"
        "add            t3,     t3,     t1      \n\t"
        "vle.v          v3,     (t3)            \n\t"
        "vfadd.vv       v19,    v0,     v1      \n\t"
        "vfadd.vv       v19,    v19,    v2      \n\t"
        "vfsub.vv       v23,    v1,     v2      \n\t"
        "vfadd.vv       v23,    v23,    v3      \n\t"
        "add            t0,     t0,     t2      \n\t"

        "mv             t0,     %[dst]          \n\t"
        "mv             t1,     %[dst_offset]   \n\t"
        "mv             t2,     %[h_offset]     \n\t"
        "mv             t3,     %[w_offset]     \n\t"
        "mv             t4,     %[dst_h]        \n\t"
        "mv             t5,     %[dst_w]        \n\t"

        "vle.v          v8,     (%[bias])       \n\t"
        "bge            t2,     t4,     END     \n\t"
        "mv             t6,     t3              \n\t"
        "bge            t6,     t5,     L1      \n\t"
        "vfadd.vv       v0,     v16,    v17     \n\t"
        "vfadd.vv       v0,     v0,     v18     \n\t"
        "vfadd.vv       v0,     v0,     v8      \n\t"
        "vse.v          v0,     (t0)            \n\t"
        "addi           s2,     t0,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     L1      \n\t"
        "vfsub.vv       v1,     v17,    v18     \n\t"
        "vfadd.vv       v1,     v1,     v19     \n\t"
        "vfadd.vv       v1,     v1,     v8      \n\t"
        "vse.v          v1,     (s2)            \n\t"

        "L1:                                    \n\t"
        "add            t0,     t0,     t1      \n\t"
        "addi           t2,     t2,     1       \n\t"
        "bge            t2,     t4,     END     \n\t"
        "mv             t6,     t3              \n\t"
        "bge            t6,     t5,     END     \n\t"
        "vfadd.vv       v0,     v20,    v21     \n\t"
        "vfadd.vv       v0,     v0,     v22     \n\t"
        "vfadd.vv       v0,     v0,     v8      \n\t"
        "vse.v          v0,     (t0)            \n\t"
        "addi           s2,     t0,     16      \n\t"

        "addi           t6,     t6,     1       \n\t"
        "bge            t6,     t5,     END     \n\t"
        "vfsub.vv       v1,     v21,    v22     \n\t"
        "vfadd.vv       v1,     v1,     v23     \n\t"
        "vfadd.vv       v1,     v1,     v8      \n\t"
        "vse.v          v1,     (s2)            \n\t"

        "END:                                   \n\t"
        "addi           x0,     x0,     1       \n\t"
        :
        : [src] "r"(dst_trans), [dst] "r"(dst), [src_offset0] "r"(dst_trans_wg_tile_stride * 8), [src_offset1] "r"(dst_trans_wg_tile_stride * 2), [dst_offset] "r"(dst_h_stride * 2), [h_offset] "r"(dst_h_offset), [w_offset] "r"(dst_w_offset), [dst_h] "r"(dst_trans_h), [dst_w] "r"(dst_trans_w), [bias] "r"(bias)
        : "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "v8");
}

ppl::common::RetCode conv2d_n8cx_wg_b2f3_fp16_runtime_executor::execute()
{
    const conv2d_common_param& cp = *conv_param_;

    LOG(DEBUG) << "n8cx wg b2f3: execute";
    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const __fp16 trans_mat_src[4][4] = {
        {1.0f, 0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 1.0f}};

    const __fp16 trans_mat_dst[2][4] = {{1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, -1.0f, 1.0f}};

    const __fp16* trans_mat_src_ = (const __fp16*)trans_mat_src;
    const __fp16* trans_mat_dst_ = (const __fp16*)trans_mat_dst;

    conv_shell_riscv_fp16<
        conv2d_n8cx_wg_bxfxs1_fp16_vec128_extra_param,
        8,
        get_real_filter_size<2, 3>,
        conv_wg_bxfxs1_riscv_per_group_fp16<2, 3, wg_b2f3s1_src_trans_kernel, wg_b2f3s1_dst_trans_kernel>>(
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
