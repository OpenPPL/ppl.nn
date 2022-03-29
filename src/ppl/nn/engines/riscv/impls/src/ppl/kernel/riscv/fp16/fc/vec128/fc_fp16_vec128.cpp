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

#include <new>
#include <cstring>

#include "ppl/kernel/riscv/fp16/fc/vec128/fc_fp16_vec128.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)8)

void fc_cvt_flt_riscv_fp16(
    const __fp16* flt,
    __fp16* flt_cvt,

    int32_t num_outs,
    int32_t channels)
{
    int32_t padded_channels = round_up(channels, 8);
    int32_t padded_num_outs = round_up(num_outs, 8);

    int32_t padded_num_outs_div8 = padded_num_outs / 8;
    for (int32_t oc = 0; oc < padded_num_outs_div8; oc++) {
        for (int32_t ic = 0; ic < padded_channels; ic++) {
            for (int32_t k = 0; k < 8; k++) {
                int32_t dst_idx = 0;
                dst_idx += k + ic * 8 + oc * padded_channels * 8;
                int32_t oc_idx  = oc * 8 + k;
                int32_t src_idx = 0;
                src_idx += ic + oc_idx * channels;
                if (oc_idx >= num_outs || ic >= channels)
                    flt_cvt[dst_idx] = 0;
                else
                    flt_cvt[dst_idx] = flt[src_idx];
            }
        }
    }
}

template <int64_t atom_m>
void hgemm_n8chw_mxn8_riscv_fp16(
    const __fp16* src,
    const __fp16* flt,
    const __fp16* bias,
    __fp16* dst,

    int32_t channels, // padded
    int32_t num_outs // padded
)
{
    asm volatile(
        ".equ           ATOM_M, %c[ATOM_M]      \n\t"

        "addi           t0,     zero,   8       \n\t"
        "vsetvli        t1,     t0,     e16     \n\t"

        "mv             t0,     %[SRC]          \n\t"
        "mv             t1,     %[FLT]          \n\t"
        "mv             t2,     %[DST]          \n\t"
        "mv             t3,     %[IC]           \n\t"

        // load bias : v31  &&  bias operation
        "vle.v          v31,    (%[BIAS])       \n\t"

        "vmv.v.v        v16,    v31             \n\t"
        ".if ATOM_M > 1                         \n\t"
        "vmv.v.v        v17,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 2                         \n\t"
        "vmv.v.v        v18,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 3                         \n\t"
        "vmv.v.v        v19,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 4                         \n\t"
        "vmv.v.v        v20,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 5                         \n\t"
        "vmv.v.v        v21,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 6                         \n\t"
        "vmv.v.v        v22,    v31             \n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 7                         \n\t"
        "vmv.v.v        v23,    v31             \n\t"
        ".endif                                 \n\t"
        // load filter : v8 - v15
        "0:                                     \n\t"
        "vle.v          v8,     (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v9,     (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v10,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v11,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v12,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v13,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v14,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v15,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"

        // load src : v0 - v7
        "mv             t4,     t0              \n\t"
        "vle.v          v0,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".if ATOM_M > 1                         \n\t"
        "vle.v          v1,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 2                         \n\t"
        "vle.v          v2,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 3                         \n\t"
        "vle.v          v3,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 4                         \n\t"
        "vle.v          v4,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 5                         \n\t"
        "vle.v          v5,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 6                         \n\t"
        "vle.v          v6,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 7                         \n\t"
        "vle.v          v7,     (t4)            \n\t"
        "add            t4,     t4,     %[IHSTD]\n\t"
        ".endif                                 \n\t"

        // calculate
        "vrgather.vi    v24,    v0,     0       \n\t"
        "vrgather.vi    v25,    v0,     1       \n\t"
        "vrgather.vi    v26,    v0,     2       \n\t"
        "vrgather.vi    v27,    v0,     3       \n\t"
        "vrgather.vi    v28,    v0,     4       \n\t"
        "vrgather.vi    v29,    v0,     5       \n\t"
        "vrgather.vi    v30,    v0,     6       \n\t"
        "vrgather.vi    v31,    v0,     7       \n\t"

        "vfmacc.vv      v16,    v8,     v24     \n\t"
        "vfmacc.vv      v16,    v9,     v25     \n\t"
        "vfmacc.vv      v16,    v10,    v26     \n\t"
        "vfmacc.vv      v16,    v11,    v27     \n\t"
        "vfmacc.vv      v16,    v12,    v28     \n\t"
        "vfmacc.vv      v16,    v13,    v29     \n\t"
        "vfmacc.vv      v16,    v14,    v30     \n\t"
        "vfmacc.vv      v16,    v15,    v31     \n\t"

        ".if ATOM_M > 1                         \n\t"
        "vrgather.vi    v24,    v1,     0       \n\t"
        "vrgather.vi    v25,    v1,     1       \n\t"
        "vrgather.vi    v26,    v1,     2       \n\t"
        "vrgather.vi    v27,    v1,     3       \n\t"
        "vrgather.vi    v28,    v1,     4       \n\t"
        "vrgather.vi    v29,    v1,     5       \n\t"
        "vrgather.vi    v30,    v1,     6       \n\t"
        "vrgather.vi    v31,    v1,     7       \n\t"

        "vfmacc.vv      v17,    v8,     v24     \n\t"
        "vfmacc.vv      v17,    v9,     v25     \n\t"
        "vfmacc.vv      v17,    v10,    v26     \n\t"
        "vfmacc.vv      v17,    v11,    v27     \n\t"
        "vfmacc.vv      v17,    v12,    v28     \n\t"
        "vfmacc.vv      v17,    v13,    v29     \n\t"
        "vfmacc.vv      v17,    v14,    v30     \n\t"
        "vfmacc.vv      v17,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 2                         \n\t"
        "vrgather.vi    v24,    v2,     0       \n\t"
        "vrgather.vi    v25,    v2,     1       \n\t"
        "vrgather.vi    v26,    v2,     2       \n\t"
        "vrgather.vi    v27,    v2,     3       \n\t"
        "vrgather.vi    v28,    v2,     4       \n\t"
        "vrgather.vi    v29,    v2,     5       \n\t"
        "vrgather.vi    v30,    v2,     6       \n\t"
        "vrgather.vi    v31,    v2,     7       \n\t"

        "vfmacc.vv      v18,    v8,     v24     \n\t"
        "vfmacc.vv      v18,    v9,     v25     \n\t"
        "vfmacc.vv      v18,    v10,    v26     \n\t"
        "vfmacc.vv      v18,    v11,    v27     \n\t"
        "vfmacc.vv      v18,    v12,    v28     \n\t"
        "vfmacc.vv      v18,    v13,    v29     \n\t"
        "vfmacc.vv      v18,    v14,    v30     \n\t"
        "vfmacc.vv      v18,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 3                         \n\t"
        "vrgather.vi    v24,    v3,     0       \n\t"
        "vrgather.vi    v25,    v3,     1       \n\t"
        "vrgather.vi    v26,    v3,     2       \n\t"
        "vrgather.vi    v27,    v3,     3       \n\t"
        "vrgather.vi    v28,    v3,     4       \n\t"
        "vrgather.vi    v29,    v3,     5       \n\t"
        "vrgather.vi    v30,    v3,     6       \n\t"
        "vrgather.vi    v31,    v3,     7       \n\t"

        "vfmacc.vv      v19,    v8,     v24     \n\t"
        "vfmacc.vv      v19,    v9,     v25     \n\t"
        "vfmacc.vv      v19,    v10,    v26     \n\t"
        "vfmacc.vv      v19,    v11,    v27     \n\t"
        "vfmacc.vv      v19,    v12,    v28     \n\t"
        "vfmacc.vv      v19,    v13,    v29     \n\t"
        "vfmacc.vv      v19,    v14,    v30     \n\t"
        "vfmacc.vv      v19,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 4                         \n\t"
        "vrgather.vi    v24,    v4,     0       \n\t"
        "vrgather.vi    v25,    v4,     1       \n\t"
        "vrgather.vi    v26,    v4,     2       \n\t"
        "vrgather.vi    v27,    v4,     3       \n\t"
        "vrgather.vi    v28,    v4,     4       \n\t"
        "vrgather.vi    v29,    v4,     5       \n\t"
        "vrgather.vi    v30,    v4,     6       \n\t"
        "vrgather.vi    v31,    v4,     7       \n\t"

        "vfmacc.vv      v20,    v8,     v24     \n\t"
        "vfmacc.vv      v20,    v9,     v25     \n\t"
        "vfmacc.vv      v20,    v10,    v26     \n\t"
        "vfmacc.vv      v20,    v11,    v27     \n\t"
        "vfmacc.vv      v20,    v12,    v28     \n\t"
        "vfmacc.vv      v20,    v13,    v29     \n\t"
        "vfmacc.vv      v20,    v14,    v30     \n\t"
        "vfmacc.vv      v20,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 5                         \n\t"
        "vrgather.vi    v24,    v5,     0       \n\t"
        "vrgather.vi    v25,    v5,     1       \n\t"
        "vrgather.vi    v26,    v5,     2       \n\t"
        "vrgather.vi    v27,    v5,     3       \n\t"
        "vrgather.vi    v28,    v5,     4       \n\t"
        "vrgather.vi    v29,    v5,     5       \n\t"
        "vrgather.vi    v30,    v5,     6       \n\t"
        "vrgather.vi    v31,    v5,     7       \n\t"

        "vfmacc.vv      v21,    v8,     v24     \n\t"
        "vfmacc.vv      v21,    v9,     v25     \n\t"
        "vfmacc.vv      v21,    v10,    v26     \n\t"
        "vfmacc.vv      v21,    v11,    v27     \n\t"
        "vfmacc.vv      v21,    v12,    v28     \n\t"
        "vfmacc.vv      v21,    v13,    v29     \n\t"
        "vfmacc.vv      v21,    v14,    v30     \n\t"
        "vfmacc.vv      v21,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 6                         \n\t"
        "vrgather.vi    v24,    v6,     0       \n\t"
        "vrgather.vi    v25,    v6,     1       \n\t"
        "vrgather.vi    v26,    v6,     2       \n\t"
        "vrgather.vi    v27,    v6,     3       \n\t"
        "vrgather.vi    v28,    v6,     4       \n\t"
        "vrgather.vi    v29,    v6,     5       \n\t"
        "vrgather.vi    v30,    v6,     6       \n\t"
        "vrgather.vi    v31,    v6,     7       \n\t"

        "vfmacc.vv      v22,    v8,     v24     \n\t"
        "vfmacc.vv      v22,    v9,     v25     \n\t"
        "vfmacc.vv      v22,    v10,    v26     \n\t"
        "vfmacc.vv      v22,    v11,    v27     \n\t"
        "vfmacc.vv      v22,    v12,    v28     \n\t"
        "vfmacc.vv      v22,    v13,    v29     \n\t"
        "vfmacc.vv      v22,    v14,    v30     \n\t"
        "vfmacc.vv      v22,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 7                         \n\t"
        "vrgather.vi    v24,    v7,     0       \n\t"
        "vrgather.vi    v25,    v7,     1       \n\t"
        "vrgather.vi    v26,    v7,     2       \n\t"
        "vrgather.vi    v27,    v7,     3       \n\t"
        "vrgather.vi    v28,    v7,     4       \n\t"
        "vrgather.vi    v29,    v7,     5       \n\t"
        "vrgather.vi    v30,    v7,     6       \n\t"
        "vrgather.vi    v31,    v7,     7       \n\t"

        "vfmacc.vv      v23,    v8,     v24     \n\t"
        "vfmacc.vv      v23,    v9,     v25     \n\t"
        "vfmacc.vv      v23,    v10,    v26     \n\t"
        "vfmacc.vv      v23,    v11,    v27     \n\t"
        "vfmacc.vv      v23,    v12,    v28     \n\t"
        "vfmacc.vv      v23,    v13,    v29     \n\t"
        "vfmacc.vv      v23,    v14,    v30     \n\t"
        "vfmacc.vv      v23,    v15,    v31     \n\t"
        ".endif                                 \n\t"

        // loop_k condition
        "addi           t0,     t0,     16      \n\t"
        "addi           t3,     t3,     -8      \n\t"
        "bne            t3,     zero,   0b      \n\t"

        // store dst : v16 - v23
        "vse.v          v16,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".if ATOM_M > 1                         \n\t"
        "vse.v          v17,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 2                         \n\t"
        "vse.v          v18,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 3                         \n\t"
        "vse.v          v19,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 4                         \n\t"
        "vse.v          v20,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 5                         \n\t"
        "vse.v          v21,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 6                         \n\t"
        "vse.v          v22,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"
        ".if ATOM_M > 7                         \n\t"
        "vse.v          v23,    (t2)            \n\t"
        "add            t2,     t2,     %[OHSTD]\n\t"
        ".endif                                 \n\t"

        :
        : [ATOM_M] "i"(atom_m), [SRC] "r"(src), [FLT] "r"(flt), [DST] "r"(dst), [BIAS] "r"(bias), [IC] "r"(channels), [IHSTD] "r"(channels * 2), [OHSTD] "r"(num_outs * 2)
        : "memory", "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

typedef void (*hgemm_n8chw_kernel_riscv_fp16_func)(const __fp16*, const __fp16*, const __fp16*, __fp16*, int32_t, int32_t);
static const hgemm_n8chw_kernel_riscv_fp16_func hgemm_n8chw_mxn8_kernel_select[8]{
    hgemm_n8chw_mxn8_riscv_fp16<1>,
    hgemm_n8chw_mxn8_riscv_fp16<2>,
    hgemm_n8chw_mxn8_riscv_fp16<3>,
    hgemm_n8chw_mxn8_riscv_fp16<4>,
    hgemm_n8chw_mxn8_riscv_fp16<5>,
    hgemm_n8chw_mxn8_riscv_fp16<6>,
    hgemm_n8chw_mxn8_riscv_fp16<7>,
    hgemm_n8chw_mxn8_riscv_fp16<8>};

void fc_n8chw_riscv_fp16(
    const __fp16* src,
    const __fp16* flt,
    const __fp16* bias,
    __fp16* dst,

    const int32_t batch,
    const int32_t num_outs,
    const int32_t channels)
{
    int32_t padded_channels = (channels + 8 - 1) / 8 * 8;
    int32_t padded_num_outs = (num_outs + 8 - 1) / 8 * 8;

    for (int32_t oc = 0; oc < padded_num_outs; oc += 8) {
        int32_t bc = 0;
        for (; bc + 8 < batch; bc += 8) {
            hgemm_n8chw_mxn8_kernel_select[7](
                src + padded_channels * bc,
                flt + padded_channels * oc,
                bias + oc,
                dst + padded_num_outs * bc + oc,

                padded_channels,
                padded_num_outs);
        }
        if (bc < batch) {
            hgemm_n8chw_mxn8_kernel_select[batch - bc - 1](
                src + padded_channels * bc,
                flt + padded_channels * oc,
                bias + oc,
                dst + padded_num_outs * bc + oc,

                padded_channels,
                padded_num_outs);
        }
    }
}

void fc_fp16_vec128_executor::cal_kernel_tunning_param()
{
    tunning_param_.m_blk = 16;
    tunning_param_.n_blk = 16;
    tunning_param_.k_blk = fc_param_->channels;
}

uint64_t fc_fp16_vec128_executor::cal_temp_buffer_size()
{
    LOG(DEBUG) << "FC cal_temp_buffer_size";
    constexpr int64_t atom_oc = 8;
    constexpr int64_t atom_ic = 8;

    tunning_param_.m_blk = min(tunning_param_.m_blk, src_shape_->GetDim(0));
    tunning_param_.n_blk = min(tunning_param_.n_blk, fc_param_->num_output);
    tunning_param_.k_blk = min(tunning_param_.k_blk, fc_param_->channels);

    return fc_common_cal_temp_buffer_size<__fp16, atom_oc, atom_ic>(
        src_shape_->GetDim(0), // m
        fc_param_->num_output, // n
        fc_param_->channels, // k
        tunning_param_);
}

ppl::common::RetCode fc_fp16_vec128_executor::prepare()
{
    if (!fc_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    cal_kernel_tunning_param();
    LOG(DEBUG) << "FC prepare";

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp16_vec128_executor::execute()
{
    if (!fc_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    LOG(DEBUG) << "FC execute";

    tunning_param_.m_blk = min(tunning_param_.m_blk, src_shape_->GetDim(0));
    tunning_param_.n_blk = min(tunning_param_.n_blk, fc_param_->num_output);
    tunning_param_.k_blk = min(tunning_param_.k_blk, fc_param_->channels);

    constexpr int64_t atom_oc = 8;
    constexpr int64_t atom_ic = 8;
    fc_common_blocking_execute<__fp16, atom_oc, atom_ic>(
        src_,
        cvt_filter_,
        cvt_bias_,
        dst_,
        temp_buffer_,
        src_shape_->GetDim(0),
        fc_param_->channels,
        fc_param_->num_output,
        tunning_param_,
        fc_n8chw_riscv_fp16);

    return common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp16_vec128_manager::gen_cvt_weights(const __fp16* filter, const __fp16* bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t padded_oc = round_up(param_.num_output, 8);
    {
        cvt_bias_size_ = padded_oc;
        cvt_bias_      = (__fp16*)allocator_->Alloc(cvt_bias_size_ * sizeof(__fp16));
        if (cvt_bias_ == nullptr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }
        memcpy(cvt_bias_, bias, param_.num_output * sizeof(__fp16));
        memset(cvt_bias_ + param_.num_output, 0, (padded_oc - param_.num_output) * sizeof(__fp16));
    }

    {
        const int32_t padded_ic = round_up(param_.channels, 8);
        cvt_filter_size_        = padded_ic * padded_oc * sizeof(__fp16);
        cvt_filter_             = (__fp16*)allocator_->Alloc(cvt_filter_size_);
        if (cvt_filter_ == nullptr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }
        fc_common_cvt_flt_to_nxcx<__fp16, C_BLK()>(filter, cvt_filter_, param_.num_output, param_.channels);
    }

    return ppl::common::RC_SUCCESS;
}

fc_executor<__fp16>* fc_fp16_vec128_manager::gen_executor()
{
    return new fc_fp16_vec128_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::riscv
