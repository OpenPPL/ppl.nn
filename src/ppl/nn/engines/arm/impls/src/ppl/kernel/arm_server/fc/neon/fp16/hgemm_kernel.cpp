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

#ifdef PPLNN_USE_ARMV8_2_FP16

#include "hgemm_kernel.h"

#include <arm_neon.h>
#include <iostream>
#include <cstdlib>

#include "ppl/kernel/arm_server/common/internal_include.h"

#define N_BLOCK0() 1
    #define M_BLOCK0() 8
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 4
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define N_BLOCK0() 2
    #define M_BLOCK0() 8
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 4
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define N_BLOCK0() 3
    #define M_BLOCK0() 4
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define N_BLOCK0() 4
    #define M_BLOCK0() 4
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 2
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
    #define M_BLOCK0() 1
        #include "hgemm_kernel_core.inc"
    #undef M_BLOCK0
#undef N_BLOCK0

#define NUM_BLOCK() 16
#define VBLOCK() 8
#define PACK_VBLOCK(val) ((val + 7) & (~7))
#define ALIGN_VBLOCK(val) ((val) & (~7))

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

size_t ppl_arm_server_kernel_fp16_fc_get_converted_filter_size(
    const int64_t num_in,
    const int64_t num_out)
{
    return PACK_VBLOCK(num_out) * (num_in + 1) * sizeof(__fp16) + 128;
}

size_t ppl_arm_server_kernel_fp16_fc_get_buffer_size(
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3)
{
    size_t out_buf_size = num_batch * sgemm_n1 * sizeof(__fp16) + 128;
    size_t in_buf_size = (num_batch > 1) ? (num_batch * sgemm_k3 * sizeof(__fp16) + 128) : 0;
    return in_buf_size + out_buf_size;
}

#define V_TRANSPOSE_FP16_8x8(v) \
    do { \
        float16x8x2_t vpf16[4]; \
        float32x4x2_t vpf32[4]; \
        vpf16[0] = vtrnq_f16(v[0], v[1]); \
        vpf16[1] = vtrnq_f16(v[2], v[3]); \
        vpf16[2] = vtrnq_f16(v[4], v[5]); \
        vpf16[3] = vtrnq_f16(v[6], v[7]); \
        vpf32[0] = vtrnq_f32((float32x4_t)vpf16[0].val[0], (float32x4_t)vpf16[1].val[0]); \
        vpf32[1] = vtrnq_f32((float32x4_t)vpf16[0].val[1], (float32x4_t)vpf16[1].val[1]); \
        vpf32[2] = vtrnq_f32((float32x4_t)vpf16[2].val[0], (float32x4_t)vpf16[3].val[0]); \
        vpf32[3] = vtrnq_f32((float32x4_t)vpf16[2].val[1], (float32x4_t)vpf16[3].val[1]); \
        v[0] = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[2].val[0])); \
        v[1] = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[0]), vget_low_f32(vpf32[3].val[0])); \
        v[2] = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[2].val[1])); \
        v[3] = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[1]), vget_low_f32(vpf32[3].val[1])); \
        v[4] = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[2].val[0])); \
        v[5] = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[0]), vget_high_f32(vpf32[3].val[0])); \
        v[6] = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[2].val[1])); \
        v[7] = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[1]), vget_high_f32(vpf32[3].val[1])); \
    } while(0)

void ppl_arm_server_kernel_fp16_fc_convert_weights(
    const __fp16 *weights,
    __fp16 *cvt_weights,
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
        for (int64_t oc = num_out; oc < CEIL8(num_out); oc++) {
            cvt_weights[ num_out_align*num_in + ic*(num_out_tail_blocks*VBLOCK()) + oc-num_out_align ] = 0.0f;
        }
    }

}

static void ppl_arm_server_kernel_fp16_fc_single_batch(
    const __fp16 *cvt_weights,
    const __fp16 *cvt_bias,
    const __fp16 *input,
    __fp16 *output,
    __fp16 *tmp_buffer,
    const int64_t num_in,
    const int64_t num_out)
{
    const int64_t num_out_align_16block = num_out & (~(int64_t)(NUM_BLOCK() * VBLOCK()-1));
    const int64_t num_out_tail_16block = num_out - num_out_align_16block;

    const int64_t num_in_align = num_in & (~(VBLOCK()-1));

PRAGMA_OMP_PARALLEL()
{
    const int64_t b_n_stride = VBLOCK();

    PRAGMA_OMP_FOR_NOWAIT()
    for (int64_t j_l1 = 0; j_l1 < num_out_align_16block; j_l1 += (NUM_BLOCK() * VBLOCK())) {
        float16x8_t vc[16];
        const __fp16 *bias_base = cvt_bias + j_l1;
        vc[0]  = vld1q_f16(bias_base + 0  * VBLOCK());
        vc[1]  = vld1q_f16(bias_base + 1  * VBLOCK());
        vc[2]  = vld1q_f16(bias_base + 2  * VBLOCK());
        vc[3]  = vld1q_f16(bias_base + 3  * VBLOCK());
        vc[4]  = vld1q_f16(bias_base + 4  * VBLOCK());
        vc[5]  = vld1q_f16(bias_base + 5  * VBLOCK());
        vc[6]  = vld1q_f16(bias_base + 6  * VBLOCK());
        vc[7]  = vld1q_f16(bias_base + 7  * VBLOCK());
        vc[8]  = vld1q_f16(bias_base + 8  * VBLOCK());
        vc[9]  = vld1q_f16(bias_base + 9  * VBLOCK());
        vc[10] = vld1q_f16(bias_base + 10 * VBLOCK());
        vc[11] = vld1q_f16(bias_base + 11 * VBLOCK());
        vc[12] = vld1q_f16(bias_base + 12 * VBLOCK());
        vc[13] = vld1q_f16(bias_base + 13 * VBLOCK());
        vc[14] = vld1q_f16(bias_base + 14 * VBLOCK());
        vc[15] = vld1q_f16(bias_base + 15 * VBLOCK());
        for (int64_t p_l1 = 0; p_l1 < num_in_align; p_l1 += VBLOCK()) {
            float16x8_t va = vld1q_f16(input + p_l1);
            float16x8_t vb0[4];
            float16x8_t vb1[4];
            const __fp16 *weights_base = cvt_weights + j_l1 * num_in + p_l1 * NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);
            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 0);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 0);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 0);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 0);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 0);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 0);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 0);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 0);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 0);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 0);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 0);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 0);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 0);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 0);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 0);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 0);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);
            
            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 1);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 1);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 1);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 1);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 1);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 1);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 1);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 1);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 1);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 1);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 1);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 1);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 1);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 1);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 1);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 1);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);

            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 2);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 2);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 2);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 2);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 2);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 2);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 2);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 2);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 2);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 2);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 2);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 2);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);
            
            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 2);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 2);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 2);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 2);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);

            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 3);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 3);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 3);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 3);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 3);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 3);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 3);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 3);
            
            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 3);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 3);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 3);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 3);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 3);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 3);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 3);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 3);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);

            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 4);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 4);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 4);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 4);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 4);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 4);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 4);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 4);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 4);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 4);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 4);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 4);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 4);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 4);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 4);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 4);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);

            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 5);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 5);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 5);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 5);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 5);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 5);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 5);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 5);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 5);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 5);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 5);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 5);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 5);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 5);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 5);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 5);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);

            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 6);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 6);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 6);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 6);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 6);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 6);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 6);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 6);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 6);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 6);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 6);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 6);

            weights_base += NUM_BLOCK() * VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 6);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 6);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 6);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 6);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);

            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 7);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 7);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 7);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 7);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 7);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 7);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 7);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 7);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 7);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 7);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 7);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 7);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 7);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 7);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 7);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 7);
        }
        for (int64_t p_l1 = num_in_align; p_l1 < num_in; p_l1++) {
            float16x8_t va = vld1q_lane_f16(input + p_l1, va, 0);
            float16x8_t vb0[4];
            float16x8_t vb1[4];
            const __fp16 *weights_base = cvt_weights + j_l1 * num_in + p_l1 * NUM_BLOCK() * VBLOCK();
            const int64_t b_n_stride = VBLOCK();
            vb0[0] = vld1q_f16(weights_base + 0  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 1  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 2  * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 3  * b_n_stride);

            vb1[0] = vld1q_f16(weights_base + 4  * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 5  * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 6  * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 7  * b_n_stride);
            
            vc[0]  = vfmaq_laneq_f16(vc[0],  vb0[0], va, 0);
            vc[1]  = vfmaq_laneq_f16(vc[1],  vb0[1], va, 0);
            vc[2]  = vfmaq_laneq_f16(vc[2],  vb0[2], va, 0);
            vc[3]  = vfmaq_laneq_f16(vc[3],  vb0[3], va, 0);

            vb0[0] = vld1q_f16(weights_base + 8  * b_n_stride);
            vb0[1] = vld1q_f16(weights_base + 9  * b_n_stride);
            vb0[2] = vld1q_f16(weights_base + 10 * b_n_stride);
            vb0[3] = vld1q_f16(weights_base + 11 * b_n_stride);

            vc[4]  = vfmaq_laneq_f16(vc[4],  vb1[0], va, 0);
            vc[5]  = vfmaq_laneq_f16(vc[5],  vb1[1], va, 0);
            vc[6]  = vfmaq_laneq_f16(vc[6],  vb1[2], va, 0);
            vc[7]  = vfmaq_laneq_f16(vc[7],  vb1[3], va, 0);

            vb1[0] = vld1q_f16(weights_base + 12 * b_n_stride);
            vb1[1] = vld1q_f16(weights_base + 13 * b_n_stride);
            vb1[2] = vld1q_f16(weights_base + 14 * b_n_stride);
            vb1[3] = vld1q_f16(weights_base + 15 * b_n_stride);

            vc[8]  = vfmaq_laneq_f16(vc[8],  vb0[0], va, 0);
            vc[9]  = vfmaq_laneq_f16(vc[9],  vb0[1], va, 0);
            vc[10] = vfmaq_laneq_f16(vc[10], vb0[2], va, 0);
            vc[11] = vfmaq_laneq_f16(vc[11], vb0[3], va, 0);

            vc[12] = vfmaq_laneq_f16(vc[12], vb1[0], va, 0);
            vc[13] = vfmaq_laneq_f16(vc[13], vb1[1], va, 0);
            vc[14] = vfmaq_laneq_f16(vc[14], vb1[2], va, 0);
            vc[15] = vfmaq_laneq_f16(vc[15], vb1[3], va, 0);
        }
        __fp16 * output_base = output + j_l1;
        vst1q_f16(output_base + 0  * VBLOCK(), vc[0] );
        vst1q_f16(output_base + 1  * VBLOCK(), vc[1] );
        vst1q_f16(output_base + 2  * VBLOCK(), vc[2] );
        vst1q_f16(output_base + 3  * VBLOCK(), vc[3] );
        vst1q_f16(output_base + 4  * VBLOCK(), vc[4] );
        vst1q_f16(output_base + 5  * VBLOCK(), vc[5] );
        vst1q_f16(output_base + 6  * VBLOCK(), vc[6] );
        vst1q_f16(output_base + 7  * VBLOCK(), vc[7] );
        vst1q_f16(output_base + 8  * VBLOCK(), vc[8] );
        vst1q_f16(output_base + 9  * VBLOCK(), vc[9] );
        vst1q_f16(output_base + 10 * VBLOCK(), vc[10]);
        vst1q_f16(output_base + 11 * VBLOCK(), vc[11]);
        vst1q_f16(output_base + 12 * VBLOCK(), vc[12]);
        vst1q_f16(output_base + 13 * VBLOCK(), vc[13]);
        vst1q_f16(output_base + 14 * VBLOCK(), vc[14]);
        vst1q_f16(output_base + 15 * VBLOCK(), vc[15]);
    }
    PRAGMA_OMP_SINGLE()
    if (num_out_tail_16block > 0) {
        const int64_t num_out_processed = num_out_align_16block;
        const int64_t num_out_align = num_out & (~(VBLOCK()-1));
        float16x8_t vc[16];
        const int64_t num_output_blocks = DIV_CEIL((num_out - num_out_processed), VBLOCK());
        const __fp16 * bias_base = cvt_bias + num_out_processed;
        for (int64_t id = 0; id < num_output_blocks; id++) {
            vc[id] = vld1q_f16(bias_base + id * VBLOCK());
        }
        for (int64_t p_l1 = 0; p_l1 < num_in_align; p_l1 += VBLOCK()) {
            float16x8_t va = vld1q_f16(input + p_l1);
            float16x8_t vb;
            const __fp16 *weights_base = cvt_weights + num_out_processed * num_in + p_l1 * num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 0);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 1);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 2);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 3);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 4);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 5);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 6);
            }
            weights_base += num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 7);
            }
        }
        for (int64_t p_l1 = num_in_align; p_l1 < num_in; p_l1++) {
            float16x8_t va = vld1q_lane_f16(input + p_l1, va, 0);
            float16x8_t vb;
            const __fp16 *weights_base = cvt_weights + num_out_processed * num_in + p_l1 * num_output_blocks * VBLOCK();
            for (int64_t id = 0; id < num_output_blocks; id++) {
                vb     = vld1q_f16(weights_base + id * b_n_stride);
                vc[id] = vfmaq_laneq_f16(vc[id],  vb,  va, 0);
            }
        }
        __fp16 * output_base = output + num_out_processed;
        const int64_t num_output_inner_blocks = (num_out_align - num_out_processed) / VBLOCK();
        for (int64_t id = 0; id < num_output_inner_blocks; id++) {
            vst1q_f16(output_base + id * VBLOCK(), vc[id]);
        }
        const int64_t num_output_tail = num_out - num_out_align;
        if (num_output_tail > 0) {
            do {
                vst1q_lane_f16(output + num_out_align    , vc[num_output_blocks-1], 0);
                if (num_output_tail == 1) break;
                vst1q_lane_f16(output + num_out_align + 1, vc[num_output_blocks-1], 1);
                if (num_output_tail == 2) break;
                vst1q_lane_f16(output + num_out_align + 2, vc[num_output_blocks-1], 2);
                if (num_output_tail == 3) break;
                vst1q_lane_f16(output + num_out_align + 3, vc[num_output_blocks-1], 3);
                if (num_output_tail == 4) break;
                vst1q_lane_f16(output + num_out_align + 4, vc[num_output_blocks-1], 4);
                if (num_output_tail == 5) break;
                vst1q_lane_f16(output + num_out_align + 5, vc[num_output_blocks-1], 5);
                if (num_output_tail == 6) break;
                vst1q_lane_f16(output + num_out_align + 6, vc[num_output_blocks-1], 6);
            } while(0);
        }
    }
}  // omp parallel region
}

static void ppl_arm_server_kernel_fp16_fc_multi_batch(
    const __fp16 *cvt_weights,
    const __fp16 *cvt_bias,
    const __fp16 *input,
    __fp16 *output,
    __fp16 *tmp_buffer,
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

    __fp16 *a_buffer = tmp_buffer;
    __fp16 *c_buffer = tmp_buffer + num_batch * sgemm_k3 * sizeof(__fp16) + 128;

    for (int64_t p_l3 = 0; p_l3 < num_in; p_l3 += sgemm_k3) {
        const int64_t k_l3 = std::min((num_in - p_l3), sgemm_k3);

        const int64_t k_l3_align = k_l3 & ( ~(VBLOCK()-1) );
        const int64_t k_l3_tail = k_l3 - k_l3_align;
        PRAGMA_OMP_FOR_NOWAIT()
        for (int64_t p = 0; p < k_l3_align; p += VBLOCK()) {
            for (int64_t i = 0; i < num_batch; i++) {
                float16x8_t vdata = vld1q_f16(input + i * num_in + p_l3 + p);
                vst1q_f16(a_buffer + p * num_batch + i * VBLOCK(), vdata);
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
                            __fp16 * c_base_local = (use_c_buffer ? (c_buffer + (i_l2+i_l1)*ldc_local) : (output + (i_l2+i_l1)*num_out + (j_l2+j_l1)));

                            const int64_t m_l1 = std::min((m_l2 - i_l1), sgemm_m1);
                            const int64_t m_l1_align = (m_l1 / k_sgemm_m0) * k_sgemm_m0;

                            const int64_t p = p_l3 + p_l1;
                            const int64_t k_l1 = std::min((k_l3-p_l1), sgemm_k1);
                            const bool is_first_k = (p == 0);

                            for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += k_sgemm_n0) {
                                for (int64_t i_l0 = 0; i_l0 < m_l1_align; i_l0 += k_sgemm_m0) {
                                    const int64_t j = j_l0 + j_l1 + j_l2;
                                    const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)k_sgemm_n0);

                                    hgemm_kernel_func_table[3][n_l0/VBLOCK()-1](
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
                                        hgemm_kernel_func_table[2][n_l0/VBLOCK()-1](
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
                                        hgemm_kernel_func_table[1][n_l0/VBLOCK()-1](
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
                                        hgemm_kernel_func_table[0][n_l0/VBLOCK()-1](
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
                                    const __fp16 * cbuf_base = c_base_local + m_idx * ldc_local;
                                    __fp16 * output_base = output + (i_l2 + i_l1 + m_idx) * num_out + j_l2 + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align; n_idx+=VBLOCK()) {
                                        float16x8_t vdata = vld1q_f16(cbuf_base + n_idx);
                                        vst1q_f16(output_base + n_idx, vdata);
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

ppl::common::RetCode fc_fp16(
    const __fp16 *cvt_weights,
    const __fp16 *cvt_bias,
    const __fp16 *input,
    __fp16 *output,
    __fp16 *tmp_buffer,
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
        ppl_arm_server_kernel_fp16_fc_single_batch(
            cvt_weights,
            cvt_bias,
            input,
            output,
            tmp_buffer,
            num_in,
            num_out);
    }
    else {
        ppl_arm_server_kernel_fp16_fc_multi_batch(
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

#endif
