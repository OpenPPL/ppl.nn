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

#include "ppl/kernel/riscv/fp32/fc/vec128/fc_fp32_vec128.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

template <int64_t atom_m>
void hgemm_n4chw_mxn4_riscv_fp32(
    const float* src,
    const float* flt,
    const float* bias,
    float* dst,

    int32_t channels, // padded
    int32_t num_outs // padded
)
{
    asm volatile(
        ".equ           ATOM_M, %c[ATOM_M]      \n\t"

        "addi           t0,     zero,   4       \n\t"
        "vsetvli        t1,     t0,     e32     \n\t"

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
        // load filter : v8 - v11
        "0:                                     \n\t"
        "vle.v          v8,     (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v9,     (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v10,    (t1)            \n\t"
        "addi           t1,     t1,     16      \n\t"
        "vle.v          v11,    (t1)            \n\t"
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

        "vfmacc.vv      v16,    v8,     v24     \n\t"
        "vfmacc.vv      v16,    v9,     v25     \n\t"
        "vfmacc.vv      v16,    v10,    v26     \n\t"
        "vfmacc.vv      v16,    v11,    v27     \n\t"

        ".if ATOM_M > 1                         \n\t"
        "vrgather.vi    v24,    v1,     0       \n\t"
        "vrgather.vi    v25,    v1,     1       \n\t"
        "vrgather.vi    v26,    v1,     2       \n\t"
        "vrgather.vi    v27,    v1,     3       \n\t"

        "vfmacc.vv      v17,    v8,     v24     \n\t"
        "vfmacc.vv      v17,    v9,     v25     \n\t"
        "vfmacc.vv      v17,    v10,    v26     \n\t"
        "vfmacc.vv      v17,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 2                         \n\t"
        "vrgather.vi    v24,    v2,     0       \n\t"
        "vrgather.vi    v25,    v2,     1       \n\t"
        "vrgather.vi    v26,    v2,     2       \n\t"
        "vrgather.vi    v27,    v2,     3       \n\t"

        "vfmacc.vv      v18,    v8,     v24     \n\t"
        "vfmacc.vv      v18,    v9,     v25     \n\t"
        "vfmacc.vv      v18,    v10,    v26     \n\t"
        "vfmacc.vv      v18,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 3                         \n\t"
        "vrgather.vi    v24,    v3,     0       \n\t"
        "vrgather.vi    v25,    v3,     1       \n\t"
        "vrgather.vi    v26,    v3,     2       \n\t"
        "vrgather.vi    v27,    v3,     3       \n\t"

        "vfmacc.vv      v19,    v8,     v24     \n\t"
        "vfmacc.vv      v19,    v9,     v25     \n\t"
        "vfmacc.vv      v19,    v10,    v26     \n\t"
        "vfmacc.vv      v19,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 4                         \n\t"
        "vrgather.vi    v24,    v4,     0       \n\t"
        "vrgather.vi    v25,    v4,     1       \n\t"
        "vrgather.vi    v26,    v4,     2       \n\t"
        "vrgather.vi    v27,    v4,     3       \n\t"

        "vfmacc.vv      v20,    v8,     v24     \n\t"
        "vfmacc.vv      v20,    v9,     v25     \n\t"
        "vfmacc.vv      v20,    v10,    v26     \n\t"
        "vfmacc.vv      v20,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 5                         \n\t"
        "vrgather.vi    v24,    v5,     0       \n\t"
        "vrgather.vi    v25,    v5,     1       \n\t"
        "vrgather.vi    v26,    v5,     2       \n\t"
        "vrgather.vi    v27,    v5,     3       \n\t"

        "vfmacc.vv      v21,    v8,     v24     \n\t"
        "vfmacc.vv      v21,    v9,     v25     \n\t"
        "vfmacc.vv      v21,    v10,    v26     \n\t"
        "vfmacc.vv      v21,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 6                         \n\t"
        "vrgather.vi    v24,    v6,     0       \n\t"
        "vrgather.vi    v25,    v6,     1       \n\t"
        "vrgather.vi    v26,    v6,     2       \n\t"
        "vrgather.vi    v27,    v6,     3       \n\t"

        "vfmacc.vv      v22,    v8,     v24     \n\t"
        "vfmacc.vv      v22,    v9,     v25     \n\t"
        "vfmacc.vv      v22,    v10,    v26     \n\t"
        "vfmacc.vv      v22,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        ".if ATOM_M > 7                         \n\t"
        "vrgather.vi    v24,    v7,     0       \n\t"
        "vrgather.vi    v25,    v7,     1       \n\t"
        "vrgather.vi    v26,    v7,     2       \n\t"
        "vrgather.vi    v27,    v7,     3       \n\t"

        "vfmacc.vv      v23,    v8,     v24     \n\t"
        "vfmacc.vv      v23,    v9,     v25     \n\t"
        "vfmacc.vv      v23,    v10,    v26     \n\t"
        "vfmacc.vv      v23,    v11,    v27     \n\t"
        ".endif                                 \n\t"

        // loop_k condition
        "addi           t0,     t0,     16      \n\t"
        "addi           t3,     t3,     -4      \n\t"
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
        : [ATOM_M] "i"(atom_m), [SRC] "r"(src), [FLT] "r"(flt), [DST] "r"(dst), [BIAS] "r"(bias), [IC] "r"(channels), [IHSTD] "r"(channels * 4), [OHSTD] "r"(num_outs * 4)
        : "memory", "t0", "t1", "t2", "t3", "t4");
}

typedef void (*hgemm_n4chw_riscv_kernel_fp32_func)(const float*, const float*, const float*, float*, int32_t, int32_t);
static const hgemm_n4chw_riscv_kernel_fp32_func hgemm_n4chw_mxn4_kernel_select[8]{
    hgemm_n4chw_mxn4_riscv_fp32<1>,
    hgemm_n4chw_mxn4_riscv_fp32<2>,
    hgemm_n4chw_mxn4_riscv_fp32<3>,
    hgemm_n4chw_mxn4_riscv_fp32<4>,
    hgemm_n4chw_mxn4_riscv_fp32<5>,
    hgemm_n4chw_mxn4_riscv_fp32<6>,
    hgemm_n4chw_mxn4_riscv_fp32<7>,
    hgemm_n4chw_mxn4_riscv_fp32<8>};

void fc_n4chw_riscv_fp32(
    const float* src,
    const float* flt,
    const float* bias,
    float* dst,

    const int32_t batch,
    const int32_t num_outs,
    const int32_t channels)
{
    int32_t padded_channels = round_up(channels, C_BLK());
    int32_t padded_num_outs = round_up(num_outs, C_BLK());

    for (int32_t oc = 0; oc < padded_num_outs; oc += C_BLK()) {
        int32_t bc = 0;
        for (; bc + C_BLK() < batch; bc += C_BLK()) {
            hgemm_n4chw_mxn4_kernel_select[7](
                src + padded_channels * bc,
                flt + padded_channels * oc,
                bias + oc,
                dst + padded_num_outs * bc + oc,

                padded_channels,
                padded_num_outs);
        }
        if (bc < batch) {
            hgemm_n4chw_mxn4_kernel_select[batch - bc - 1](
                src + padded_channels * bc,
                flt + padded_channels * oc,
                bias + oc,
                dst + padded_num_outs * bc + oc,

                padded_channels,
                padded_num_outs);
        }
    }
}

void fc_fp32_vec128_executor::cal_kernel_tunning_param()
{
    tunning_param_.m_blk = 16;
    tunning_param_.n_blk = 16;
    tunning_param_.k_blk = fc_param_->channels;
}

uint64_t fc_fp32_vec128_executor::cal_temp_buffer_size()
{
    LOG(DEBUG) << "FC cal_temp_buffer_size";
    constexpr int64_t atom_oc = 4;
    constexpr int64_t atom_ic = 4;

    tunning_param_.m_blk = min(tunning_param_.m_blk, src_shape_->GetDim(0));
    tunning_param_.n_blk = min(tunning_param_.n_blk, fc_param_->num_output);
    tunning_param_.k_blk = min(tunning_param_.k_blk, fc_param_->channels);

    return fc_common_cal_temp_buffer_size<float, atom_oc, atom_ic>(
        src_shape_->GetDim(0), // m
        fc_param_->num_output, // n
        fc_param_->channels, // k
        tunning_param_);
}

ppl::common::RetCode fc_fp32_vec128_executor::prepare()
{
    if (!fc_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    cal_kernel_tunning_param();
    LOG(DEBUG) << "FC prepare";

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp32_vec128_executor::execute()
{
    if (!fc_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    LOG(DEBUG) << "FC execute";

    tunning_param_.m_blk = min(tunning_param_.m_blk, src_shape_->GetDim(0));
    tunning_param_.n_blk = min(tunning_param_.n_blk, fc_param_->num_output);
    tunning_param_.k_blk = min(tunning_param_.k_blk, fc_param_->channels);

    constexpr int64_t atom_oc = 4;
    constexpr int64_t atom_ic = 4;
    fc_common_blocking_execute<float, atom_oc, atom_ic>(
        src_,
        cvt_filter_,
        cvt_bias_,
        dst_,
        temp_buffer_,
        src_shape_->GetDim(0),
        fc_param_->channels,
        fc_param_->num_output,
        tunning_param_,
        fc_n4chw_riscv_fp32);

    return common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp32_vec128_manager::gen_cvt_weights(const float* filter, const float* bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t padded_oc = round_up(param_.num_output, C_BLK());
    {
        cvt_bias_size_ = padded_oc;
        cvt_bias_      = (float*)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
        if (cvt_bias_ == nullptr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }
        memcpy(cvt_bias_, bias, param_.num_output * sizeof(float));
        memset(cvt_bias_ + param_.num_output, 0, (padded_oc - param_.num_output) * sizeof(float));
    }

    {
        const int32_t padded_ic = round_up(param_.channels, C_BLK());
        cvt_filter_size_        = padded_ic * padded_oc * sizeof(float);
        cvt_filter_             = (float*)allocator_->Alloc(cvt_filter_size_);
        if (cvt_filter_ == nullptr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }
        fc_common_cvt_flt_to_nxcx<float, C_BLK()>(filter, cvt_filter_, param_.num_output, param_.channels);
    }
    return ppl::common::RC_SUCCESS;
}

fc_executor<float>* fc_fp32_vec128_manager::gen_executor()
{
    return new fc_fp32_vec128_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::riscv
