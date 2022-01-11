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

#include "sgemm_kernel.h"

#include <arm_neon.h>
#include <iostream>
#include <cstdlib>

#include "ppl/kernel/arm_server/common/internal_include.h"

#define N_BLOCK0() 1
    #define M_BLOCK0() 15
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 8
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 4
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define N_BLOCK0() 2
    #define M_BLOCK0() 10
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 8
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 4
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define N_BLOCK0() 3
    #define M_BLOCK0() 7
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 4
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define N_BLOCK0() 4
    #define M_BLOCK0() 5
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 4
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "sgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define NUM_BLOCK() 16
#define VBLOCK() 4
#define PACK_VBLOCK(val) ((val + 3) & (~3))

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

size_t ppl_arm_server_kernel_fp32_fc_get_converted_filter_size(
    const int64_t num_in,
    const int64_t num_out)
{
    return PACK_VBLOCK(num_out) * (num_in + 1) * sizeof(float) + 128;
}

size_t ppl_arm_server_kernel_fp32_fc_get_buffer_size(
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_k3)
{
    size_t out_buf_size = num_batch * sgemm_n1 * sizeof(float) + 128;
    size_t in_buf_size = (num_batch > 1) ? (num_batch * sgemm_k3 * sizeof(float) + 128) : 0;
    return in_buf_size + out_buf_size;
}

#define V_TRANSPOSE_FP32_4x4(v) \
    do { \
        float32x4x2_t vpf32[2]; \
        vpf32[0] = vtrnq_f32(v[0], v[1]); \
        vpf32[1] = vtrnq_f32(v[2], v[3]); \
        v[0] = vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[1].val[0])); \
        v[1] = vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[1].val[1])); \
        v[2] = vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[1].val[0])); \
        v[3] = vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[1].val[1])); \
    } while(0)

void ppl_arm_server_kernel_fp32_fc_convert_weights(
    const float *weights,
    float *cvt_weights,
    const int64_t num_in,
    const int64_t num_out) 
{
    const int64_t num_out_align = num_out & (~(NUM_BLOCK() * VBLOCK()-1));
    const int64_t num_out_tail  = num_out - num_out_align;
    const int64_t num_out_tail_blocks  = DIV_CEIL(num_out_tail, VBLOCK());
    for (int64_t ic = 0; ic < num_in; ic++) {
        for (int64_t oc = 0; oc < num_out_align; oc++) {
            cvt_weights[ (oc/(NUM_BLOCK()*VBLOCK())*num_in)*(NUM_BLOCK()*VBLOCK()) + ic*(NUM_BLOCK()*VBLOCK()) + oc%(NUM_BLOCK()*VBLOCK()) ] = weights[oc*num_in+ic];
        }
        for (int64_t oc = num_out_align; oc < num_out; oc++) {
            cvt_weights[ num_out_align*num_in + ic*(num_out_tail_blocks*VBLOCK()) + oc-num_out_align ] = weights[oc*num_in+ic];
        }
        for (int64_t oc = num_out; oc < CEIL4(num_out); oc++) {
            cvt_weights[ num_out_align*num_in + ic*(num_out_tail_blocks*VBLOCK()) + oc-num_out_align ] = 0.0f;
        }
    }
}

static void ppl_arm_server_kernel_fp32_fc_single_batch(
    const float *cvt_weights,
    const float *cvt_bias,
    const float *input,
    float *output,
    float *tmp_buffer,
    const int64_t num_in,
    const int64_t num_out)
{
    const int64_t num_out_align_16block = num_out & (~(int64_t)(NUM_BLOCK() * VBLOCK()-1));
    const int64_t num_out_tail_16block = num_out - num_out_align_16block;

    const int64_t num_in_align = num_in & (~(VBLOCK()-1));

PRAGMA_OMP_PARALLEL()
{
    PRAGMA_OMP_FOR_NOWAIT()
    for (int64_t j_l1 = 0; j_l1 < num_out_align_16block; j_l1 += NUM_BLOCK() * VBLOCK()) {
        float32x4_t vc[16];
        const float *bias_base = cvt_bias + j_l1;
        vc[0]  = vld1q_f32(bias_base + 0  * VBLOCK());
        vc[1]  = vld1q_f32(bias_base + 1  * VBLOCK());
        vc[2]  = vld1q_f32(bias_base + 2  * VBLOCK());
        vc[3]  = vld1q_f32(bias_base + 3  * VBLOCK());
        vc[4]  = vld1q_f32(bias_base + 4  * VBLOCK());
        vc[5]  = vld1q_f32(bias_base + 5  * VBLOCK());
        vc[6]  = vld1q_f32(bias_base + 6  * VBLOCK());
        vc[7]  = vld1q_f32(bias_base + 7  * VBLOCK());
        vc[8]  = vld1q_f32(bias_base + 8  * VBLOCK());
        vc[9]  = vld1q_f32(bias_base + 9  * VBLOCK());
        vc[10] = vld1q_f32(bias_base + 10 * VBLOCK());
        vc[11] = vld1q_f32(bias_base + 11 * VBLOCK());
        vc[12] = vld1q_f32(bias_base + 12 * VBLOCK());
        vc[13] = vld1q_f32(bias_base + 13 * VBLOCK());
        vc[14] = vld1q_f32(bias_base + 14 * VBLOCK());
        vc[15] = vld1q_f32(bias_base + 15 * VBLOCK());
        for (int64_t p_l1 = 0; p_l1 < num_in_align; p_l1 += 4) {
            float32x4_t va = vld1q_f32(input + p_l1);
            float32x4_t vb0[4];
            float32x4_t vb1[4];
            const float *weights_base = cvt_weights + j_l1 * num_in + p_l1 * NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f32(weights_base + 0  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 1  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 2  * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 3  * VBLOCK());

            vb1[0] = vld1q_f32(weights_base + 4  * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 5  * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 6  * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 7  * VBLOCK());

            vc[0]  = vfmaq_laneq_f32(vc[0],  vb0[0], va, 0);
            vc[1]  = vfmaq_laneq_f32(vc[1],  vb0[1], va, 0);
            vc[2]  = vfmaq_laneq_f32(vc[2],  vb0[2], va, 0);
            vc[3]  = vfmaq_laneq_f32(vc[3],  vb0[3], va, 0);

            vb0[0] = vld1q_f32(weights_base + 8  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 9  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 10 * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 11 * VBLOCK());

            vc[4]  = vfmaq_laneq_f32(vc[4],  vb1[0], va, 0);
            vc[5]  = vfmaq_laneq_f32(vc[5],  vb1[1], va, 0);
            vc[6]  = vfmaq_laneq_f32(vc[6],  vb1[2], va, 0);
            vc[7]  = vfmaq_laneq_f32(vc[7],  vb1[3], va, 0);

            vb1[0] = vld1q_f32(weights_base + 12 * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 13 * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 14 * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 15 * VBLOCK());

            vc[8]  = vfmaq_laneq_f32(vc[8],  vb0[0], va, 0);
            vc[9]  = vfmaq_laneq_f32(vc[9],  vb0[1], va, 0);
            vc[10] = vfmaq_laneq_f32(vc[10], vb0[2], va, 0);
            vc[11] = vfmaq_laneq_f32(vc[11], vb0[3], va, 0);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f32(weights_base + 0  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 1  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 2  * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 3  * VBLOCK());

            vc[12] = vfmaq_laneq_f32(vc[12], vb1[0], va, 0);
            vc[13] = vfmaq_laneq_f32(vc[13], vb1[1], va, 0);
            vc[14] = vfmaq_laneq_f32(vc[14], vb1[2], va, 0);
            vc[15] = vfmaq_laneq_f32(vc[15], vb1[3], va, 0);

            vb1[0] = vld1q_f32(weights_base + 4  * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 5  * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 6  * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 7  * VBLOCK());

            vc[0]  = vfmaq_laneq_f32(vc[0],  vb0[0], va, 1);
            vc[1]  = vfmaq_laneq_f32(vc[1],  vb0[1], va, 1);
            vc[2]  = vfmaq_laneq_f32(vc[2],  vb0[2], va, 1);
            vc[3]  = vfmaq_laneq_f32(vc[3],  vb0[3], va, 1);

            vb0[0] = vld1q_f32(weights_base + 8  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 9  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 10 * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 11 * VBLOCK());

            vc[4]  = vfmaq_laneq_f32(vc[4],  vb1[0], va, 1);
            vc[5]  = vfmaq_laneq_f32(vc[5],  vb1[1], va, 1);
            vc[6]  = vfmaq_laneq_f32(vc[6],  vb1[2], va, 1);
            vc[7]  = vfmaq_laneq_f32(vc[7],  vb1[3], va, 1);

            vb1[0] = vld1q_f32(weights_base + 12 * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 13 * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 14 * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 15 * VBLOCK());

            vc[8]  = vfmaq_laneq_f32(vc[8],  vb0[0], va, 1);
            vc[9]  = vfmaq_laneq_f32(vc[9],  vb0[1], va, 1);
            vc[10] = vfmaq_laneq_f32(vc[10], vb0[2], va, 1);
            vc[11] = vfmaq_laneq_f32(vc[11], vb0[3], va, 1);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f32(weights_base + 0  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 1  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 2  * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 3  * VBLOCK());

            vc[12] = vfmaq_laneq_f32(vc[12], vb1[0], va, 1);
            vc[13] = vfmaq_laneq_f32(vc[13], vb1[1], va, 1);
            vc[14] = vfmaq_laneq_f32(vc[14], vb1[2], va, 1);
            vc[15] = vfmaq_laneq_f32(vc[15], vb1[3], va, 1);

            vb1[0] = vld1q_f32(weights_base + 4  * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 5  * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 6  * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 7  * VBLOCK());

            vc[0]  = vfmaq_laneq_f32(vc[0],  vb0[0], va, 2);
            vc[1]  = vfmaq_laneq_f32(vc[1],  vb0[1], va, 2);
            vc[2]  = vfmaq_laneq_f32(vc[2],  vb0[2], va, 2);
            vc[3]  = vfmaq_laneq_f32(vc[3],  vb0[3], va, 2);

            vb0[0] = vld1q_f32(weights_base + 8  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 9  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 10 * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 11 * VBLOCK());

            vc[4]  = vfmaq_laneq_f32(vc[4],  vb1[0], va, 2);
            vc[5]  = vfmaq_laneq_f32(vc[5],  vb1[1], va, 2);
            vc[6]  = vfmaq_laneq_f32(vc[6],  vb1[2], va, 2);
            vc[7]  = vfmaq_laneq_f32(vc[7],  vb1[3], va, 2);

            vb1[0] = vld1q_f32(weights_base + 12 * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 13 * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 14 * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 15 * VBLOCK());

            vc[8]  = vfmaq_laneq_f32(vc[8],  vb0[0], va, 2);
            vc[9]  = vfmaq_laneq_f32(vc[9],  vb0[1], va, 2);
            vc[10] = vfmaq_laneq_f32(vc[10], vb0[2], va, 2);
            vc[11] = vfmaq_laneq_f32(vc[11], vb0[3], va, 2);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f32(weights_base + 0  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 1  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 2  * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 3  * VBLOCK());

            vc[12] = vfmaq_laneq_f32(vc[12], vb1[0], va, 2);
            vc[13] = vfmaq_laneq_f32(vc[13], vb1[1], va, 2);
            vc[14] = vfmaq_laneq_f32(vc[14], vb1[2], va, 2);
            vc[15] = vfmaq_laneq_f32(vc[15], vb1[3], va, 2);

            vb1[0] = vld1q_f32(weights_base + 4  * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 5  * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 6  * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 7  * VBLOCK());

            vc[0]  = vfmaq_laneq_f32(vc[0],  vb0[0], va, 3);
            vc[1]  = vfmaq_laneq_f32(vc[1],  vb0[1], va, 3);
            vc[2]  = vfmaq_laneq_f32(vc[2],  vb0[2], va, 3);
            vc[3]  = vfmaq_laneq_f32(vc[3],  vb0[3], va, 3);

            vb0[0] = vld1q_f32(weights_base + 8  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 9  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 10 * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 11 * VBLOCK());

            vc[4]  = vfmaq_laneq_f32(vc[4],  vb1[0], va, 3);
            vc[5]  = vfmaq_laneq_f32(vc[5],  vb1[1], va, 3);
            vc[6]  = vfmaq_laneq_f32(vc[6],  vb1[2], va, 3);
            vc[7]  = vfmaq_laneq_f32(vc[7],  vb1[3], va, 3);

            vb1[0] = vld1q_f32(weights_base + 12 * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 13 * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 14 * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 15 * VBLOCK());

            vc[8]  = vfmaq_laneq_f32(vc[8],  vb0[0], va, 3);
            vc[9]  = vfmaq_laneq_f32(vc[9],  vb0[1], va, 3);
            vc[10] = vfmaq_laneq_f32(vc[10], vb0[2], va, 3);
            vc[11] = vfmaq_laneq_f32(vc[11], vb0[3], va, 3);

            vc[12] = vfmaq_laneq_f32(vc[12], vb1[0], va, 3);
            vc[13] = vfmaq_laneq_f32(vc[13], vb1[1], va, 3);
            vc[14] = vfmaq_laneq_f32(vc[14], vb1[2], va, 3);
            vc[15] = vfmaq_laneq_f32(vc[15], vb1[3], va, 3);
        }
        for (int64_t p_l1 = num_in_align; p_l1 < num_in; p_l1++) {
            float32x4_t va = vld1q_lane_f32(input + p_l1, va, 0);
            float32x4_t vb0[4];
            float32x4_t vb1[4];
            const float *weights_base = cvt_weights + j_l1 * num_in + p_l1 * NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f32(weights_base + 0  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 1  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 2  * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 3  * VBLOCK());

            vb1[0] = vld1q_f32(weights_base + 4  * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 5  * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 6  * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 7  * VBLOCK());

            vc[0]  = vfmaq_laneq_f32(vc[0],  vb0[0], va, 0);
            vc[1]  = vfmaq_laneq_f32(vc[1],  vb0[1], va, 0);
            vc[2]  = vfmaq_laneq_f32(vc[2],  vb0[2], va, 0);
            vc[3]  = vfmaq_laneq_f32(vc[3],  vb0[3], va, 0);

            vb0[0] = vld1q_f32(weights_base + 8  * VBLOCK());
            vb0[1] = vld1q_f32(weights_base + 9  * VBLOCK());
            vb0[2] = vld1q_f32(weights_base + 10 * VBLOCK());
            vb0[3] = vld1q_f32(weights_base + 11 * VBLOCK());

            vc[4]  = vfmaq_laneq_f32(vc[4],  vb1[0], va, 0);
            vc[5]  = vfmaq_laneq_f32(vc[5],  vb1[1], va, 0);
            vc[6]  = vfmaq_laneq_f32(vc[6],  vb1[2], va, 0);
            vc[7]  = vfmaq_laneq_f32(vc[7],  vb1[3], va, 0);

            vb1[0] = vld1q_f32(weights_base + 12 * VBLOCK());
            vb1[1] = vld1q_f32(weights_base + 13 * VBLOCK());
            vb1[2] = vld1q_f32(weights_base + 14 * VBLOCK());
            vb1[3] = vld1q_f32(weights_base + 15 * VBLOCK());

            vc[8]  = vfmaq_laneq_f32(vc[8],  vb0[0], va, 0);
            vc[9]  = vfmaq_laneq_f32(vc[9],  vb0[1], va, 0);
            vc[10] = vfmaq_laneq_f32(vc[10], vb0[2], va, 0);
            vc[11] = vfmaq_laneq_f32(vc[11], vb0[3], va, 0);

            vc[12] = vfmaq_laneq_f32(vc[12], vb1[0], va, 0);
            vc[13] = vfmaq_laneq_f32(vc[13], vb1[1], va, 0);
            vc[14] = vfmaq_laneq_f32(vc[14], vb1[2], va, 0);
            vc[15] = vfmaq_laneq_f32(vc[15], vb1[3], va, 0);
        }
        float * output_base = output + j_l1;
        vst1q_f32(output_base + 0 , vc[0] );
        vst1q_f32(output_base + 4 , vc[1] );
        vst1q_f32(output_base + 8 , vc[2] );
        vst1q_f32(output_base + 12, vc[3] );
        vst1q_f32(output_base + 16, vc[4] );
        vst1q_f32(output_base + 20, vc[5] );
        vst1q_f32(output_base + 24, vc[6] );
        vst1q_f32(output_base + 28, vc[7] );
        vst1q_f32(output_base + 32, vc[8] );
        vst1q_f32(output_base + 36, vc[9] );
        vst1q_f32(output_base + 40, vc[10]);
        vst1q_f32(output_base + 44, vc[11]);
        vst1q_f32(output_base + 48, vc[12]);
        vst1q_f32(output_base + 52, vc[13]);
        vst1q_f32(output_base + 56, vc[14]);
        vst1q_f32(output_base + 60, vc[15]);
    }
    PRAGMA_OMP_SINGLE()
    if (num_out_tail_16block > 0) {
        const int64_t num_out_processed = num_out_align_16block;
        const int64_t num_out_align = num_out & (~(VBLOCK()-1));
        float32x4_t vc[16];
        const int64_t num_output_blocks = DIV_CEIL((num_out - num_out_processed), VBLOCK());
        const float * bias_base = cvt_bias + num_out_processed;
        for (int64_t id = 0; id < num_output_blocks; id++) {
            vc[id] = vld1q_f32(bias_base + id * VBLOCK());
        }
        for (int64_t p_l1 = 0; p_l1 < num_in_align; p_l1 += VBLOCK()) {
            float32x4_t va = vld1q_f32(input + p_l1);
            float32x4_t vb;
            const float *weights_base = cvt_weights + num_out_processed * num_in + p_l1 * num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f32(weights_base + id * VBLOCK());
                vc[id]  = vfmaq_laneq_f32(vc[id],  vb,  va, 0);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f32(weights_base + id * VBLOCK());
                vc[id]  = vfmaq_laneq_f32(vc[id],  vb,  va, 1);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f32(weights_base + id * VBLOCK());
                vc[id]  = vfmaq_laneq_f32(vc[id],  vb,  va, 2);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f32(weights_base + id * VBLOCK());
                vc[id]  = vfmaq_laneq_f32(vc[id],  vb,  va, 3);
            }
        }
        for (int64_t p_l1 = num_in_align; p_l1 < num_in; p_l1++) {
            float32x4_t va = vld1q_lane_f32(input + p_l1, va, 0);
            float32x4_t vb;
            const float *weights_base = cvt_weights + num_out_processed * num_in + p_l1 * num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f32(weights_base + id * VBLOCK());
                vc[id]  = vfmaq_laneq_f32(vc[id],  vb,  va, 0);
            }
        }
        float * output_base = output + num_out_processed;
        const int64_t num_output_inner_blocks = (num_out_align - num_out_processed) / 4;
        for (int64_t id = 0; id < num_output_inner_blocks; id++) {
            vst1q_f32(output_base + id * 4, vc[id]);
        }
        const int64_t num_output_tail = num_out - num_out_align;
        if (num_output_tail > 0) {
            do {
                vst1q_lane_f32(output + num_out_align, vc[num_output_blocks-1], 0);
                if (num_output_tail == 1) break;
                vst1q_lane_f32(output + num_out_align + 1, vc[num_output_blocks-1], 1);
                if (num_output_tail == 2) break;
                vst1q_lane_f32(output + num_out_align + 2, vc[num_output_blocks-1], 2);
            } while (0);
        }
    }
}  // omp parallel region
}

static void ppl_arm_server_kernel_fp32_fc_multi_batch(
    const float *cvt_weights,
    const float *cvt_bias,
    const float *input,
    float *output,
    float *tmp_buffer,
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m2,
    const int64_t sgemm_n2,
    const int64_t sgemm_k3)
{

    int64_t opt_sgemm_m2 = sgemm_m2;
    int64_t opt_sgemm_n2 = sgemm_n2;
    
PRAGMA_OMP_PARALLEL()
{
    const int64_t k_sgemm_m0 = 8;
    const int64_t k_sgemm_n0 = VBLOCK() * 2;

    float *a_buffer = tmp_buffer;
    float *c_buffer = tmp_buffer + num_batch * sgemm_k3 * sizeof(float) + 128;

    for (int64_t p_l3 = 0; p_l3 < num_in; p_l3 += sgemm_k3) {
        const int64_t k_l3 = std::min((num_in - p_l3), sgemm_k3);

        const int64_t k_l3_align = k_l3 & ( ~(VBLOCK()-1) );
        const int64_t k_l3_tail = k_l3 - k_l3_align;
        PRAGMA_OMP_FOR_NOWAIT()
        for (int64_t p = 0; p < k_l3_align; p += VBLOCK()) {
            for (int64_t i = 0; i < num_batch; i++) {
                float32x4_t vdata = vld1q_f32(input + i * num_in + p_l3 + p);
                vst1q_f32(a_buffer + p * num_batch + i * VBLOCK(), vdata);
            }
        }
        PRAGMA_OMP_SINGLE()
        if (k_l3_tail) {
            for (int64_t i = 0; i < num_batch; i++) {
                for (int64_t p = 0; p < k_l3_tail; p++) {
                    a_buffer[k_l3_align * num_batch + i * VBLOCK() + p] = input[i * num_in + p_l3 + k_l3_align + p];
                }
                for (int64_t p = k_l3_tail; p < VBLOCK(); p++) {
                    a_buffer[k_l3_align * num_batch + i * VBLOCK() + p] = 0.0f;
                }
            }
        }
        // PRAGMA_OMP_BARRIER()

        PRAGMA_OMP_FOR_COLLAPSE(2)
        for (int64_t j_l2 = 0; j_l2 < num_out; j_l2 += opt_sgemm_n2) {
            for (int64_t i_l2 = 0; i_l2 < num_batch; i_l2 += opt_sgemm_m2) {
                const int64_t m_l2 = std::min((num_batch-i_l2), opt_sgemm_m2);
                const int64_t n_l2 = std::min((num_out-j_l2), opt_sgemm_n2);

                for (int64_t j_l1 = 0; j_l1 < n_l2; j_l1 += sgemm_n1) {
                    const int64_t n_multi_vblock_id = (j_l2 + j_l1) / (NUM_BLOCK() * VBLOCK());
                    const int64_t n_multi_vblock_num = std::min((int64_t)NUM_BLOCK(), PACK_VBLOCK(num_out)/VBLOCK() - n_multi_vblock_id * NUM_BLOCK());
                    const int64_t j_ofs = (j_l1 % (NUM_BLOCK() * VBLOCK()));

                    for (int64_t p_l1 = 0; p_l1 < k_l3; p_l1 += sgemm_k1) {
                        for (int64_t i_l1 = 0; i_l1 < m_l2; i_l1 += sgemm_m1) {
                            const int64_t n_l1 = std::min((n_l2 - j_l1), sgemm_n1);
                            const int64_t n_l1_pack = PACK_VBLOCK(n_l1);
                            const bool use_c_buffer = (n_l1 % VBLOCK());
                            const int64_t ldc_local = (use_c_buffer ? n_l1_pack : num_out);
                            float * c_base_local = (use_c_buffer ? (c_buffer + (i_l2+i_l1)*ldc_local) : (output + (i_l2+i_l1)*num_out + (j_l2+j_l1)));

                            const int64_t m_l1 = std::min((m_l2 - i_l1), sgemm_m1);
                            const int64_t m_l1_align = (m_l1 / k_sgemm_m0) * k_sgemm_m0;

                            const int64_t p = p_l3 + p_l1;
                            const int64_t k_l1 = std::min((k_l3-p_l1), sgemm_k1);
                            const bool is_first_k = (p == 0);

                            for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += k_sgemm_n0) {
                                for (int64_t i_l0 = 0; i_l0 < m_l1_align; i_l0 += k_sgemm_m0) {
                                    const int64_t j = j_l0 + j_l1 + j_l2;
                                    const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)k_sgemm_n0);

                                    sgemm_kernel_func_table[3][n_l0/VBLOCK()-1](
                                        a_buffer + p_l1 * num_batch + (i_l2+i_l1+i_l0) * VBLOCK(),
                                        cvt_weights + n_multi_vblock_id * num_in * (NUM_BLOCK() * VBLOCK()) + p * n_multi_vblock_num * VBLOCK() + j_ofs + j_l0,
                                        cvt_bias + j,
                                        c_base_local + i_l0 * ldc_local + j_l0,
                                        k_sgemm_m0,
                                        n_l0,
                                        k_l1,
                                        VBLOCK(),
                                        num_batch * VBLOCK(),
                                        n_multi_vblock_num * VBLOCK(), // b_k
                                        VBLOCK(), // b_n
                                        ldc_local,
                                        !is_first_k,
                                        0
                                    );
                                }
                            }
                            const int64_t m_l1_tail = m_l1 - m_l1_align;
                            if (m_l1_tail) {
                                for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += VBLOCK() * 4) {
                                    int64_t i_l0 = m_l1_align;
                                    int64_t i = i_l0 + i_l1 + i_l2;
                                    const int64_t j = j_l0 + j_l1 + j_l2;
                                    if (m_l1_tail & 4) {
                                        const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)VBLOCK() * 4);
                                        sgemm_kernel_func_table[2][n_l0/VBLOCK()-1](
                                            a_buffer + p_l1 * num_batch + (i_l2+i_l1+i_l0) * VBLOCK(),
                                            cvt_weights + n_multi_vblock_id * num_in * (NUM_BLOCK() * VBLOCK()) + p * n_multi_vblock_num * VBLOCK() + j_ofs + j_l0,
                                            cvt_bias + j,
                                            c_base_local + i_l0 * ldc_local + j_l0,
                                            4,
                                            n_l0,
                                            k_l1,
                                            VBLOCK(),
                                            num_batch * VBLOCK(),
                                            n_multi_vblock_num * VBLOCK(), // b_k
                                            VBLOCK(), // b_n
                                            ldc_local,
                                            !is_first_k,
                                            0
                                        );
                                        i_l0 += 4;
                                        i += 4;
                                    }
                                    if (m_l1_tail & 2) {
                                        const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)VBLOCK() * 4);
                                        sgemm_kernel_func_table[1][n_l0/VBLOCK()-1](
                                            a_buffer + p_l1 * num_batch + (i_l2+i_l1+i_l0) * VBLOCK(),
                                            cvt_weights + n_multi_vblock_id * num_in * (NUM_BLOCK() * VBLOCK()) + p * n_multi_vblock_num * VBLOCK() + j_ofs + j_l0,
                                            cvt_bias + j,
                                            c_base_local + i_l0 * ldc_local + j_l0,
                                            2,
                                            n_l0,
                                            k_l1,
                                            VBLOCK(),
                                            num_batch * VBLOCK(),
                                            n_multi_vblock_num * VBLOCK(), // b_k
                                            VBLOCK(), // b_n
                                            ldc_local,
                                            !is_first_k,
                                            0
                                        );
                                        i_l0 += 2;
                                        i += 2;
                                    }
                                    if (m_l1_tail & 1) {
                                        const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)VBLOCK() * 4);
                                        sgemm_kernel_func_table[0][n_l0/VBLOCK()-1](
                                            a_buffer + p_l1 * num_batch + (i_l2+i_l1+i_l0) * VBLOCK(),
                                            cvt_weights + n_multi_vblock_id * num_in * (NUM_BLOCK() * VBLOCK()) + p * n_multi_vblock_num * VBLOCK() + j_ofs + j_l0,
                                            cvt_bias + j,
                                            c_base_local + i_l0 * ldc_local + j_l0,
                                            1,
                                            n_l0,
                                            k_l1,
                                            VBLOCK(),
                                            num_batch * VBLOCK(),
                                            n_multi_vblock_num * VBLOCK(), // b_k
                                            VBLOCK(), // b_n
                                            ldc_local,
                                            !is_first_k,
                                            0
                                        );
                                    }
                                }
                            }

                            if (use_c_buffer && (p_l3 + sgemm_k3 >= num_in)) {

                                const int64_t n_l1_align = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const float * cbuf_base = c_base_local + m_idx * ldc_local;
                                    float * output_base = output + (i_l2 + i_l1 + m_idx) * num_out + j_l2 + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align; n_idx+=VBLOCK()) {
                                        float32x4_t vdata = vld1q_f32(cbuf_base + n_idx);
                                        vst1q_f32(output_base + n_idx, vdata);
                                    }
                                    for (int64_t n_idx = n_l1_align; n_idx < n_l1; n_idx++) {
                                        output_base[n_idx] = cbuf_base[n_idx];
                                    }
                                }
                            }
                        }  // M_l1
                    }  // K_l1
                }  // N_l1
            }  // M_l2
        }  // N_l2
    }  // K_l3
}  // omp parallel region
}

ppl::common::RetCode fc_fp32(
    const float *cvt_weights,
    const float *cvt_bias,
    const float *input,
    float *output,
    float *tmp_buffer,
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m2,
    const int64_t sgemm_n2,
    const int64_t sgemm_k3)
{
    if (num_batch == 1) {
        ppl_arm_server_kernel_fp32_fc_single_batch(
            cvt_weights,
            cvt_bias,
            input,
            output,
            tmp_buffer,
            num_in,
            num_out);
    }
    else {
        ppl_arm_server_kernel_fp32_fc_multi_batch(
            cvt_weights,
            cvt_bias,
            input,
            output,
            tmp_buffer,
            num_in,
            num_out,
            num_batch,
            sgemm_m1,
            sgemm_n1,
            sgemm_k1,
            sgemm_m2,
            sgemm_n2,
            sgemm_k3);
    }
    return ppl::common::RC_SUCCESS;
}

}}}}  // namespace ppl::kernel::arm_server::neon
