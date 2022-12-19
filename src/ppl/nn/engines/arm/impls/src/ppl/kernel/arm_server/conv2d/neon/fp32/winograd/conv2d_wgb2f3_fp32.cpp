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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/winograd/conv2d_wgb2f3_fp32.h"

#include <arm_neon.h>
#include <chrono>
#include <cstring>
#include <malloc.h>

#include "ppl/kernel/arm_server/conv2d/neon/fp32/n4cx_sgemm/n4cx_sgemm.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define CBLK()  4
#define ICBLK() CBLK()
#define OCBLK() CBLK()

#define WINOGRAD_B2F3_OUTPUT_BLOCKSIZE() 2
#define WINOGRAD_B2F3_INPUT_BLOCKSIZE()  4
#define WINOGRAD_B2F3_NUM_SET()          16

#define WGB2F3_OBLK() WINOGRAD_B2F3_OUTPUT_BLOCKSIZE()
#define WGB2F3_IBLK() WINOGRAD_B2F3_INPUT_BLOCKSIZE()
#define WGB2F3_NSET() WINOGRAD_B2F3_NUM_SET()

#define N4CX_SGEMM_N10_BLOCK0() 10
#define N4CX_SGEMM_N12_BLOCK0() 12

#define LLC_CACHELINE_SIZE() 128

size_t conv2d_n4cx_wgb2f3_get_input_buffer_size_fp32(
    const int64_t channels,
    const int64_t tile_l2_size)
{
    /* inner parallel mode */
    const int64_t num_wg_blocks    = tile_l2_size;
    const size_t input_buffer_size = CEIL128(WGB2F3_NSET() * CEIL4(channels) * num_wg_blocks * sizeof(float)) + LLC_CACHELINE_SIZE();
    return input_buffer_size;
}

static size_t conv2d_n4cx_wgb2f3_get_output_buffer_size_fp32(
    const int64_t num_output,
    const int64_t tile_l2_size)
{
    /* inner parallel mode */
    const int64_t num_wg_blocks     = tile_l2_size;
    const size_t output_buffer_size = CEIL128(WGB2F3_NSET() * CEIL4(num_output) * num_wg_blocks * sizeof(float)) + LLC_CACHELINE_SIZE();
    return output_buffer_size;
}

static inline void conv2d_n4cx_wgb2f3_prep_input_block_fp32(
    const float *input_block,
    const float32x4_t &vzeros,
    float *prep_input_block,
    const int64_t src_w,
    const int64_t in_wg_set_offset,
    const bool ih_valid[WGB2F3_IBLK()],
    const bool iw_valid[WGB2F3_IBLK()])
{
    float32x4_t v[16];

    // D[0][:]
    if (ih_valid[0]) {
        const float *input_base = input_block;
        v[0]                    = iw_valid[0] ? vld1q_f32(input_base) : vzeros;
        v[1]                    = iw_valid[1] ? vld1q_f32(input_base + 1 * ICBLK()) : vzeros;
        v[2]                    = iw_valid[2] ? vld1q_f32(input_base + 2 * ICBLK()) : vzeros;
        v[3]                    = iw_valid[3] ? vld1q_f32(input_base + 3 * ICBLK()) : vzeros;
    } else {
        v[0] = vzeros;
        v[1] = vzeros;
        v[2] = vzeros;
        v[3] = vzeros;
    }

    // D[2][:]
    if (ih_valid[2]) {
        const float *input_base = input_block + 2 * src_w * ICBLK();
        v[4]                    = iw_valid[0] ? vld1q_f32(input_base) : vzeros;
        v[5]                    = iw_valid[1] ? vld1q_f32(input_base + 1 * ICBLK()) : vzeros;
        v[6]                    = iw_valid[2] ? vld1q_f32(input_base + 2 * ICBLK()) : vzeros;
        v[7]                    = iw_valid[3] ? vld1q_f32(input_base + 3 * ICBLK()) : vzeros;
    } else {
        v[4] = vzeros;
        v[5] = vzeros;
        v[6] = vzeros;
        v[7] = vzeros;
    }

    // B_t[0][:] : D[0][:], D[2][:]
    // (B_t * D)[0][:] <- B_t[0][:] * D[:][:]
    v[8]  = vsubq_f32(v[0], v[4]);
    v[9]  = vsubq_f32(v[1], v[5]);
    v[10] = vsubq_f32(v[2], v[6]);
    v[11] = vsubq_f32(v[3], v[7]);
    // (B_t * D * B)[0][:] <- (B_t * D)[0][:] * B[:][:]
    v[12] = vsubq_f32(v[8], v[10]);
    v[13] = vaddq_f32(v[9], v[10]);
    v[14] = vsubq_f32(v[10], v[9]);
    v[15] = vsubq_f32(v[9], v[11]);

    vst1q_f32(prep_input_block + 0 * in_wg_set_offset, v[12]);
    vst1q_f32(prep_input_block + 1 * in_wg_set_offset, v[13]);
    vst1q_f32(prep_input_block + 2 * in_wg_set_offset, v[14]);
    vst1q_f32(prep_input_block + 3 * in_wg_set_offset, v[15]);

    // D[1][:]
    if (ih_valid[1]) {
        const float *input_base = input_block + 1 * src_w * ICBLK();
        v[0]                    = iw_valid[0] ? vld1q_f32(input_base) : vzeros;
        v[1]                    = iw_valid[1] ? vld1q_f32(input_base + 1 * ICBLK()) : vzeros;
        v[2]                    = iw_valid[2] ? vld1q_f32(input_base + 2 * ICBLK()) : vzeros;
        v[3]                    = iw_valid[3] ? vld1q_f32(input_base + 3 * ICBLK()) : vzeros;
    } else {
        v[0] = vzeros;
        v[1] = vzeros;
        v[2] = vzeros;
        v[3] = vzeros;
    }

    // B_t[1][:] : D[1][:], D[2][:]
    // (B_t * D)[1][:] <- B_t[1][:] * D[:][:]
    v[8]  = vaddq_f32(v[0], v[4]);
    v[9]  = vaddq_f32(v[1], v[5]);
    v[10] = vaddq_f32(v[2], v[6]);
    v[11] = vaddq_f32(v[3], v[7]);
    // (B_t * D * B)[1][:] <- (B_t * D)[1][:] * B[:][:]
    v[12] = vsubq_f32(v[8], v[10]);
    v[13] = vaddq_f32(v[9], v[10]);
    v[14] = vsubq_f32(v[10], v[9]);
    v[15] = vsubq_f32(v[9], v[11]);

    vst1q_f32(prep_input_block + 4 * in_wg_set_offset, v[12]);
    vst1q_f32(prep_input_block + 5 * in_wg_set_offset, v[13]);
    vst1q_f32(prep_input_block + 6 * in_wg_set_offset, v[14]);
    vst1q_f32(prep_input_block + 7 * in_wg_set_offset, v[15]);

    // B_t[2][:] : D[1][:], D[2][:]
    // (B_t * D)[2][:] <- B_t[2][:] * D[:][:]
    v[8]  = vsubq_f32(v[4], v[0]);
    v[9]  = vsubq_f32(v[5], v[1]);
    v[10] = vsubq_f32(v[6], v[2]);
    v[11] = vsubq_f32(v[7], v[3]);
    // (B_t * D * B)[2][:] <- (B_t * D)[2][:] * B[:][:]
    v[12] = vsubq_f32(v[8], v[10]);
    v[13] = vaddq_f32(v[9], v[10]);
    v[14] = vsubq_f32(v[10], v[9]);
    v[15] = vsubq_f32(v[9], v[11]);

    vst1q_f32(prep_input_block + 8 * in_wg_set_offset, v[12]);
    vst1q_f32(prep_input_block + 9 * in_wg_set_offset, v[13]);
    vst1q_f32(prep_input_block + 10 * in_wg_set_offset, v[14]);
    vst1q_f32(prep_input_block + 11 * in_wg_set_offset, v[15]);

    // D[3][:]
    if (ih_valid[3]) {
        const float *input_base = input_block + 3 * src_w * ICBLK();
        v[4]                    = iw_valid[0] ? vld1q_f32(input_base) : vzeros;
        v[5]                    = iw_valid[1] ? vld1q_f32(input_base + 1 * ICBLK()) : vzeros;
        v[6]                    = iw_valid[2] ? vld1q_f32(input_base + 2 * ICBLK()) : vzeros;
        v[7]                    = iw_valid[3] ? vld1q_f32(input_base + 3 * ICBLK()) : vzeros;
    } else {
        v[4] = vzeros;
        v[5] = vzeros;
        v[6] = vzeros;
        v[7] = vzeros;
    }

    // B_t[3][:] : D[1][:], D[3][:]
    // (B_t * D)[3][:] <- B_t[3][:] * D[:][:]
    v[8]  = vsubq_f32(v[0], v[4]);
    v[9]  = vsubq_f32(v[1], v[5]);
    v[10] = vsubq_f32(v[2], v[6]);
    v[11] = vsubq_f32(v[3], v[7]);
    // (B_t * D * B)[3][:] <- (B_t * D)[3][:] * B[:][:]
    v[12] = vsubq_f32(v[8], v[10]);
    v[13] = vaddq_f32(v[9], v[10]);
    v[14] = vsubq_f32(v[10], v[9]);
    v[15] = vsubq_f32(v[9], v[11]);

    vst1q_f32(prep_input_block + 12 * in_wg_set_offset, v[12]);
    vst1q_f32(prep_input_block + 13 * in_wg_set_offset, v[13]);
    vst1q_f32(prep_input_block + 14 * in_wg_set_offset, v[14]);
    vst1q_f32(prep_input_block + 15 * in_wg_set_offset, v[15]);
}

static inline void conv2d_n4cx_wgb2f3_postp_output_block_fp32(
    const float *raw_output_block,
    const float32x4_t &vbias,
    float *output_block, // oc_start biased, oh_start, ow_start biased
    float *sum_block,
    const int64_t wg_out_set_offset,
    const int64_t dst_w,
    const int64_t num_valid_oh,
    const int64_t num_valid_ow,
    const conv_fuse_flag_t fuse_flag)
{
    float32x4_t vr[16];
    if (num_valid_oh == 2 && num_valid_ow == 2) {
        vr[0]  = vld1q_f32(raw_output_block + 0 * wg_out_set_offset);
        vr[1]  = vld1q_f32(raw_output_block + 1 * wg_out_set_offset);
        vr[2]  = vld1q_f32(raw_output_block + 2 * wg_out_set_offset);
        vr[3]  = vld1q_f32(raw_output_block + 3 * wg_out_set_offset);
        vr[4]  = vld1q_f32(raw_output_block + 4 * wg_out_set_offset);
        vr[5]  = vld1q_f32(raw_output_block + 5 * wg_out_set_offset);
        vr[6]  = vld1q_f32(raw_output_block + 6 * wg_out_set_offset);
        vr[7]  = vld1q_f32(raw_output_block + 7 * wg_out_set_offset);
        vr[8]  = vld1q_f32(raw_output_block + 8 * wg_out_set_offset);
        vr[9]  = vld1q_f32(raw_output_block + 9 * wg_out_set_offset);
        vr[10] = vld1q_f32(raw_output_block + 10 * wg_out_set_offset);
        vr[11] = vld1q_f32(raw_output_block + 11 * wg_out_set_offset);
        vr[12] = vld1q_f32(raw_output_block + 12 * wg_out_set_offset);
        vr[13] = vld1q_f32(raw_output_block + 13 * wg_out_set_offset);
        vr[14] = vld1q_f32(raw_output_block + 14 * wg_out_set_offset);
        vr[15] = vld1q_f32(raw_output_block + 15 * wg_out_set_offset);

        vr[0] = vaddq_f32(vr[0], vr[4]);
        vr[1] = vaddq_f32(vr[1], vr[5]);
        vr[2] = vaddq_f32(vr[2], vr[6]);
        vr[3] = vaddq_f32(vr[3], vr[7]);

        vr[4] = vsubq_f32(vr[4], vr[8]);
        vr[5] = vsubq_f32(vr[5], vr[9]);
        vr[6] = vsubq_f32(vr[6], vr[10]);
        vr[7] = vsubq_f32(vr[7], vr[11]);

        vr[0] = vaddq_f32(vr[0], vr[8]);
        vr[1] = vaddq_f32(vr[1], vr[9]);
        vr[2] = vaddq_f32(vr[2], vr[10]);
        vr[3] = vaddq_f32(vr[3], vr[11]);

        vr[4] = vsubq_f32(vr[4], vr[12]);
        vr[5] = vsubq_f32(vr[5], vr[13]);
        vr[6] = vsubq_f32(vr[6], vr[14]);
        vr[7] = vsubq_f32(vr[7], vr[15]);

        vr[0] = vaddq_f32(vr[0], vr[1]);
        vr[4] = vaddq_f32(vr[4], vr[5]);
        vr[1] = vsubq_f32(vr[1], vr[2]);
        vr[5] = vsubq_f32(vr[5], vr[6]);

        vr[0] = vaddq_f32(vr[0], vr[2]);
        vr[4] = vaddq_f32(vr[4], vr[6]);
        vr[1] = vsubq_f32(vr[1], vr[3]);
        vr[5] = vsubq_f32(vr[5], vr[7]);

        vr[0] = vaddq_f32(vr[0], vbias);
        vr[1] = vaddq_f32(vr[1], vbias);
        vr[4] = vaddq_f32(vr[4], vbias);
        vr[5] = vaddq_f32(vr[5], vbias);

        if (fuse_flag & conv_fuse_flag::SUM) { // sum
            vr[0] = vaddq_f32(vr[0], vld1q_f32(sum_block));
            vr[1] = vaddq_f32(vr[1], vld1q_f32(sum_block + OCBLK()));
            vr[4] = vaddq_f32(vr[4], vld1q_f32(sum_block + dst_w * OCBLK()));
            vr[5] = vaddq_f32(vr[5], vld1q_f32(sum_block + dst_w * OCBLK() + OCBLK()));
        }

        if (fuse_flag & conv_fuse_flag::RELU) { // relu
            float32x4_t vzero = vdupq_n_f32(0.0f);
            vr[0]             = vmaxq_f32(vr[0], vzero);
            vr[1]             = vmaxq_f32(vr[1], vzero);
            vr[4]             = vmaxq_f32(vr[4], vzero);
            vr[5]             = vmaxq_f32(vr[5], vzero);
        }

        if (fuse_flag & conv_fuse_flag::RELU6) { // relu6
            float32x4_t vsix = vdupq_n_f32(6.0f);
            vr[0]            = vminq_f32(vr[0], vsix);
            vr[1]            = vminq_f32(vr[1], vsix);
            vr[4]            = vminq_f32(vr[4], vsix);
            vr[5]            = vminq_f32(vr[5], vsix);
        }

        vst1q_f32(output_block, vr[0]);
        vst1q_f32(output_block + OCBLK(), vr[1]);
        vst1q_f32(output_block + dst_w * OCBLK(), vr[4]);
        vst1q_f32(output_block + dst_w * OCBLK() + OCBLK(), vr[5]);
    } else if (num_valid_oh == 2 && num_valid_ow == 1) {
        vr[0]  = vld1q_f32(raw_output_block + 0 * wg_out_set_offset);
        vr[1]  = vld1q_f32(raw_output_block + 1 * wg_out_set_offset);
        vr[2]  = vld1q_f32(raw_output_block + 2 * wg_out_set_offset);
        vr[4]  = vld1q_f32(raw_output_block + 4 * wg_out_set_offset);
        vr[5]  = vld1q_f32(raw_output_block + 5 * wg_out_set_offset);
        vr[6]  = vld1q_f32(raw_output_block + 6 * wg_out_set_offset);
        vr[8]  = vld1q_f32(raw_output_block + 8 * wg_out_set_offset);
        vr[9]  = vld1q_f32(raw_output_block + 9 * wg_out_set_offset);
        vr[10] = vld1q_f32(raw_output_block + 10 * wg_out_set_offset);
        vr[12] = vld1q_f32(raw_output_block + 12 * wg_out_set_offset);
        vr[13] = vld1q_f32(raw_output_block + 13 * wg_out_set_offset);
        vr[14] = vld1q_f32(raw_output_block + 14 * wg_out_set_offset);

        vr[0] = vaddq_f32(vr[0], vr[4]);
        vr[1] = vaddq_f32(vr[1], vr[5]);
        vr[2] = vaddq_f32(vr[2], vr[6]);

        vr[4] = vsubq_f32(vr[4], vr[8]);
        vr[5] = vsubq_f32(vr[5], vr[9]);
        vr[6] = vsubq_f32(vr[6], vr[10]);

        vr[0] = vaddq_f32(vr[0], vr[8]);
        vr[1] = vaddq_f32(vr[1], vr[9]);
        vr[2] = vaddq_f32(vr[2], vr[10]);

        vr[4] = vsubq_f32(vr[4], vr[12]);
        vr[5] = vsubq_f32(vr[5], vr[13]);
        vr[6] = vsubq_f32(vr[6], vr[14]);

        vr[0] = vaddq_f32(vr[0], vr[1]);
        vr[4] = vaddq_f32(vr[4], vr[5]);

        vr[0] = vaddq_f32(vr[0], vr[2]);
        vr[4] = vaddq_f32(vr[4], vr[6]);

        vr[0] = vaddq_f32(vr[0], vbias);
        vr[4] = vaddq_f32(vr[4], vbias);

        if (fuse_flag & conv_fuse_flag::SUM) { // sum
            vr[0] = vaddq_f32(vr[0], vld1q_f32(sum_block));
            vr[4] = vaddq_f32(vr[4], vld1q_f32(sum_block + dst_w * OCBLK()));
        }

        if (fuse_flag & conv_fuse_flag::RELU) { // relu
            float32x4_t vzero = vdupq_n_f32(0.0f);
            vr[0]             = vmaxq_f32(vr[0], vzero);
            vr[4]             = vmaxq_f32(vr[4], vzero);
        }

        if (fuse_flag & conv_fuse_flag::RELU6) { // relu6
            float32x4_t vsix = vdupq_n_f32(6.0f);
            vr[0]            = vminq_f32(vr[0], vsix);
            vr[4]            = vminq_f32(vr[4], vsix);
        }

        vst1q_f32(output_block, vr[0]);
        vst1q_f32(output_block + dst_w * OCBLK(), vr[4]);
    } else if (num_valid_oh == 1 && num_valid_ow == 2) {
        vr[0]  = vld1q_f32(raw_output_block + 0 * wg_out_set_offset);
        vr[1]  = vld1q_f32(raw_output_block + 1 * wg_out_set_offset);
        vr[2]  = vld1q_f32(raw_output_block + 2 * wg_out_set_offset);
        vr[3]  = vld1q_f32(raw_output_block + 3 * wg_out_set_offset);
        vr[4]  = vld1q_f32(raw_output_block + 4 * wg_out_set_offset);
        vr[5]  = vld1q_f32(raw_output_block + 5 * wg_out_set_offset);
        vr[6]  = vld1q_f32(raw_output_block + 6 * wg_out_set_offset);
        vr[7]  = vld1q_f32(raw_output_block + 7 * wg_out_set_offset);
        vr[8]  = vld1q_f32(raw_output_block + 8 * wg_out_set_offset);
        vr[9]  = vld1q_f32(raw_output_block + 9 * wg_out_set_offset);
        vr[10] = vld1q_f32(raw_output_block + 10 * wg_out_set_offset);
        vr[11] = vld1q_f32(raw_output_block + 11 * wg_out_set_offset);

        vr[0] = vaddq_f32(vr[0], vr[4]);
        vr[1] = vaddq_f32(vr[1], vr[5]);
        vr[2] = vaddq_f32(vr[2], vr[6]);
        vr[3] = vaddq_f32(vr[3], vr[7]);

        vr[0] = vaddq_f32(vr[0], vr[8]);
        vr[1] = vaddq_f32(vr[1], vr[9]);
        vr[2] = vaddq_f32(vr[2], vr[10]);
        vr[3] = vaddq_f32(vr[3], vr[11]);

        vr[0] = vaddq_f32(vr[0], vr[1]);
        vr[1] = vsubq_f32(vr[1], vr[2]);

        vr[0] = vaddq_f32(vr[0], vr[2]);
        vr[1] = vsubq_f32(vr[1], vr[3]);

        vr[0] = vaddq_f32(vr[0], vbias);
        vr[1] = vaddq_f32(vr[1], vbias);

        if (fuse_flag & conv_fuse_flag::SUM) { // sum
            vr[0] = vaddq_f32(vr[0], vld1q_f32(sum_block));
            vr[1] = vaddq_f32(vr[1], vld1q_f32(sum_block + OCBLK()));
        }

        if (fuse_flag & conv_fuse_flag::RELU) { // relu
            float32x4_t vzero = vdupq_n_f32(0.0f);
            vr[0]             = vmaxq_f32(vr[0], vzero);
            vr[1]             = vmaxq_f32(vr[1], vzero);
        }

        if (fuse_flag & conv_fuse_flag::RELU6) { // relu6
            float32x4_t vsix = vdupq_n_f32(6.0f);
            vr[0]            = vminq_f32(vr[0], vsix);
            vr[1]            = vminq_f32(vr[1], vsix);
        }

        vst1q_f32(output_block, vr[0]);
        vst1q_f32(output_block + OCBLK(), vr[1]);
    } else if (num_valid_oh == 1 && num_valid_ow == 1) {
        vr[0]  = vld1q_f32(raw_output_block + 0 * wg_out_set_offset);
        vr[1]  = vld1q_f32(raw_output_block + 1 * wg_out_set_offset);
        vr[2]  = vld1q_f32(raw_output_block + 2 * wg_out_set_offset);
        vr[4]  = vld1q_f32(raw_output_block + 4 * wg_out_set_offset);
        vr[5]  = vld1q_f32(raw_output_block + 5 * wg_out_set_offset);
        vr[6]  = vld1q_f32(raw_output_block + 6 * wg_out_set_offset);
        vr[8]  = vld1q_f32(raw_output_block + 8 * wg_out_set_offset);
        vr[9]  = vld1q_f32(raw_output_block + 9 * wg_out_set_offset);
        vr[10] = vld1q_f32(raw_output_block + 10 * wg_out_set_offset);

        vr[0] = vaddq_f32(vr[0], vr[4]);
        vr[1] = vaddq_f32(vr[1], vr[5]);
        vr[2] = vaddq_f32(vr[2], vr[6]);

        vr[0] = vaddq_f32(vr[0], vr[8]);
        vr[1] = vaddq_f32(vr[1], vr[9]);
        vr[2] = vaddq_f32(vr[2], vr[10]);

        vr[0] = vaddq_f32(vr[0], vr[1]);
        vr[0] = vaddq_f32(vr[0], vr[2]);

        vr[0] = vaddq_f32(vr[0], vbias);

        if (fuse_flag & conv_fuse_flag::SUM) { // sum
            vr[0] = vaddq_f32(vr[0], vld1q_f32(sum_block));
        }

        if (fuse_flag & conv_fuse_flag::RELU) { // relu
            float32x4_t vzero = vdupq_n_f32(0.0f);
            vr[0]             = vmaxq_f32(vr[0], vzero);
        }

        if (fuse_flag & conv_fuse_flag::RELU6) { // relu6
            float32x4_t vsix = vdupq_n_f32(6.0f);
            vr[0]            = vminq_f32(vr[0], vsix);
        }

        vst1q_f32(output_block, vr[0]);
    }
}

uint64_t conv2d_wgb2f3_fp32_runtime_executor::cal_temp_buffer_size()
{
    const conv2d_param &cp                      = *conv_param_;
    const conv2d_wgb2f3_fp32_schedule_param &sp = sched_param_;
    size_t input_buffer_size                    = conv2d_n4cx_wgb2f3_get_input_buffer_size_fp32(
        cp.channels, sp.tile_seg);
    size_t output_buffer_size = conv2d_n4cx_wgb2f3_get_output_buffer_size_fp32(
        cp.num_output, sp.tile_seg);

    sched_param_.input_buffer_size  = input_buffer_size;
    sched_param_.output_buffer_size = output_buffer_size;

    size_t total_buffer_size = input_buffer_size + output_buffer_size;
    return total_buffer_size;
}

void conv2d_wgb2f3_fp32_runtime_executor::adjust_schedule_param()
{
    return;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_runtime_executor::execute()
{
    const conv2d_param &cp                      = *conv_param_;
    const conv2d_wgb2f3_fp32_schedule_param &sp = sched_param_;
    const float *input                          = (const float *)src_;
    const float *cvt_filter                     = (const float *)cvt_filter_;
    const float *bias                           = (const float *)cvt_bias_;
    float *output                               = (float *)dst_;
    float *sum                                  = (float *)sum_;
    float *tmp_buffer                           = (float *)temp_buffer_;
    const int64_t src_h                         = src_shape_->GetDim(2);
    const int64_t src_w                         = src_shape_->GetDim(3);
    const int64_t channels                      = src_shape_->GetDim(1);
    const int64_t num_output                    = cp.num_output;
    const int64_t dst_h                         = dst_shape_->GetDim(2);
    const int64_t dst_w                         = dst_shape_->GetDim(3);
    const int64_t pad_h                         = cp.pad_h;
    const int64_t pad_w                         = cp.pad_w;
    const int64_t group                         = cp.group;
    const int64_t ics                           = sp.ic_seg;
    const int64_t ocs                           = sp.oc_seg;
    const int64_t tile_l2_size                  = sp.tile_seg;
    const int64_t num_batch                     = src_shape_->GetDim(0);
    const size_t input_prep_buffer_size         = sp.input_buffer_size;

    PRAGMA_OMP_PARALLEL()
    {
        const int64_t ic_packed = CEIL4(channels);
        const int64_t oc_packed = CEIL4(num_output);

        const int64_t ic_group    = channels / group;
        const int64_t oc_group    = num_output / group;
        const int64_t ic_g_packed = CEIL4(ic_group);
        const int64_t oc_g_packed = CEIL4(oc_group);

        const int64_t k_in_channel_section  = CEIL4(std::min(ics, ic_group));
        const int64_t k_out_channel_section = CEIL4(std::min(ocs, oc_group));

        const int64_t k_tile_l2 = tile_l2_size;

        const int64_t k_in_wg_set_offset = k_in_channel_section * k_tile_l2;

        /* Inner parallel mode */
        float *pre_proc_buffer  = tmp_buffer;
        float *post_proc_buffer = pre_proc_buffer + input_prep_buffer_size / sizeof(float);

        const float32x4_t vzeros = vdupq_n_f32(0.0f);

        const int64_t num_h_blocks  = DIV_CEIL(dst_h, WGB2F3_OBLK());
        const int64_t num_w_blocks  = DIV_CEIL(dst_w, WGB2F3_OBLK());
        const int64_t num_hw_blocks = num_h_blocks * num_w_blocks;
        const int64_t num_tiles     = num_batch * num_hw_blocks;

        const int64_t hw_in               = src_h * src_w;
        const int64_t hw_out              = dst_h * dst_w;
        const int64_t input_b_stride      = ic_packed * hw_in;
        const int64_t output_b_stride     = oc_packed * hw_out;
        const int64_t input_g_stride      = ic_group * hw_in;
        const int64_t output_g_stride     = oc_group * hw_out;
        const int64_t filter_wgset_stride = oc_g_packed * ic_g_packed;
        const int64_t filter_g_stride     = WGB2F3_NSET() * filter_wgset_stride;

        for (int64_t g = 0; g < group; g++) {
            const float *input_g_base  = input + g * input_g_stride;
            const float *filter_g_base = cvt_filter + g * filter_g_stride;
            const float *bias_g_base   = bias + g * oc_group;
            float *output_g_base       = output + g * output_g_stride;
            float *sum_g_base          = sum + g * output_g_stride;

            for (int64_t tile_l2 = 0; tile_l2 < num_tiles; tile_l2 += k_tile_l2) {
                const int64_t wg_blocks = std::min(k_tile_l2, num_tiles - tile_l2);

                // Note: using `ic_group` in the loop is the same with using `ic_g_packed`.
                for (int64_t ic_l2 = 0; ic_l2 < ic_g_packed; ic_l2 += k_in_channel_section) {
                    const bool is_first_ic           = (ic_l2 == 0);
                    const bool is_last_ic            = (ic_l2 + k_in_channel_section >= ic_g_packed);
                    const int64_t in_channel_section = std::min(ic_g_packed - ic_l2, k_in_channel_section);

                    PRAGMA_OMP_FOR_COLLAPSE(2)
                    for (int64_t ic = 0; ic < in_channel_section; ic += ICBLK()) {
                        for (int64_t tile_l0 = 0; tile_l0 < wg_blocks; tile_l0++) {
                            int64_t tile_id        = tile_l0 + tile_l2;
                            const int64_t batch_id = tile_id / num_hw_blocks;

                            const int64_t tile_hw_id = tile_id % num_hw_blocks;
                            const int64_t tile_h_id  = tile_hw_id / num_w_blocks;
                            const int64_t tile_w_id  = tile_hw_id % num_w_blocks;

                            const int64_t oh = tile_h_id * WGB2F3_OBLK();
                            const int64_t ow = tile_w_id * WGB2F3_OBLK();

                            const float *input_c_base = input_g_base + batch_id * input_b_stride + (ic_l2 + ic) * hw_in;
                            float *prep_in_c_base     = pre_proc_buffer + ic * wg_blocks;

                            const int64_t ih0 = -pad_h + oh;
                            const int64_t ih1 = ih0 + 1;
                            const int64_t ih2 = ih0 + 2;
                            const int64_t ih3 = ih0 + 3;

                            bool ih_valid[WGB2F3_IBLK()];
                            ih_valid[0] = (ih0 >= 0 && ih0 < src_h);
                            ih_valid[1] = (ih1 >= 0 && ih1 < src_h);
                            ih_valid[2] = (ih2 >= 0 && ih2 < src_h);
                            ih_valid[3] = (ih3 >= 0 && ih3 < src_h);

                            int64_t wg_block_idx = tile_l0;
                            float *prep_in_block = prep_in_c_base + wg_block_idx * ICBLK();

                            const int64_t iw0 = -pad_w + ow;
                            const int64_t iw1 = iw0 + 1;
                            const int64_t iw2 = iw0 + 2;
                            const int64_t iw3 = iw0 + 3;

                            bool iw_valid[WGB2F3_IBLK()];
                            iw_valid[0] = (iw0 >= 0 && iw0 < src_w);
                            iw_valid[1] = (iw1 >= 0 && iw1 < src_w);
                            iw_valid[2] = (iw2 >= 0 && iw2 < src_w);
                            iw_valid[3] = (iw3 >= 0 && iw3 < src_w);

                            conv2d_n4cx_wgb2f3_prep_input_block_fp32(
                                input_c_base + ih0 * src_w * ICBLK() + iw0 * ICBLK(),
                                vzeros,
                                prep_in_block,
                                src_w,
                                k_in_wg_set_offset,
                                ih_valid,
                                iw_valid);
                        } // close loop over tile
                    } // close loop over ic(register)

                    const int32_t init_id = (is_first_ic) ? 0 : 2;
                    const int64_t fini_id = 0;
                    PRAGMA_OMP_FOR_COLLAPSE(2)
                    for (int64_t set_id = 0; set_id < WGB2F3_NSET(); set_id++) {
                        // Note: using `oc_group` in the loop is the same with using `oc_g_packed`.
                        for (int64_t oc_l2 = 0; oc_l2 < oc_g_packed; oc_l2 += k_out_channel_section) {
                            const int64_t out_channel_section       = std::min(oc_g_packed - oc_l2, k_out_channel_section);
                            const int64_t k_out_wg_set_local_offset = out_channel_section * wg_blocks;
                            const float *cvt_filter_cc_base         = filter_g_base + oc_l2 * ic_g_packed + ic_l2 * CEIL4(out_channel_section); // pack to 4:OCBLK()
                            float *raw_out_cl2_base                 = post_proc_buffer + WGB2F3_NSET() * oc_l2 * wg_blocks + set_id * k_out_wg_set_local_offset;

                            const int64_t out_channel_section_align_2ocblk = FLOOR8(out_channel_section);
                            for (int64_t oc = 0; oc < out_channel_section_align_2ocblk; oc += 2 * OCBLK()) {
                                for (int64_t block = 0; block < wg_blocks; block += N4CX_SGEMM_N10_BLOCK0()) {
                                    const int64_t m_l0 = std::min((int64_t)OCBLK(), out_channel_section - oc);
                                    const int64_t n_l0 = std::min((int64_t)N4CX_SGEMM_N10_BLOCK0(), wg_blocks - block);
                                    sgemm_n4cx_kernel_m8nx_fp32_func_table[n_l0 - 1][init_id][fini_id](
                                        cvt_filter_cc_base + set_id * filter_wgset_stride + oc * CEIL4(in_channel_section),
                                        pre_proc_buffer + set_id * k_in_wg_set_offset + block * ICBLK(),
                                        nullptr, /* constant:bias */
                                        nullptr, /* fusedata:sum */
                                        raw_out_cl2_base + oc * wg_blocks + block * OCBLK(),
                                        m_l0,
                                        n_l0,
                                        in_channel_section,
                                        oc_g_packed,
                                        wg_blocks,
                                        0,
                                        wg_blocks);
                                } // close loop over wg-block(register)
                            } // close loop over oc(register)
                            if (out_channel_section > out_channel_section_align_2ocblk) {
                                int64_t oc = out_channel_section_align_2ocblk;
                                for (int64_t block = 0; block < wg_blocks; block += N4CX_SGEMM_N12_BLOCK0()) {
                                    const int64_t m_l0 = std::min((int64_t)OCBLK(), out_channel_section - oc);
                                    const int64_t n_l0 = std::min((int64_t)N4CX_SGEMM_N12_BLOCK0(), wg_blocks - block);
                                    sgemm_n4cx_kernel_m4nx_fp32_func_table[n_l0 - 1][init_id][fini_id](
                                        cvt_filter_cc_base + set_id * filter_wgset_stride + oc * CEIL4(in_channel_section),
                                        pre_proc_buffer + set_id * k_in_wg_set_offset + block * ICBLK(),
                                        nullptr, /* constant:bias */
                                        nullptr, /* fusedata:sum */
                                        raw_out_cl2_base + oc * wg_blocks + block * OCBLK(),
                                        m_l0,
                                        n_l0,
                                        in_channel_section,
                                        oc_g_packed,
                                        wg_blocks,
                                        0,
                                        wg_blocks);
                                } // close loop over wg-block(register)
                            }
                        } // close loop over oc(l2)
                    } // close loop over wg-set
                    // NOTE: implicit omp barrier

                    if (is_last_ic) {
                        for (int64_t oc_l2 = 0; oc_l2 < oc_g_packed; oc_l2 += k_out_channel_section) {
                            const int64_t out_channel_section       = std::min(oc_g_packed - oc_l2, k_out_channel_section);
                            const int64_t k_out_wg_set_local_offset = out_channel_section * wg_blocks;
                            const float *raw_out_cl2_base           = post_proc_buffer + WGB2F3_NSET() * oc_l2 * wg_blocks;
                            float *output_oc_l2_base                = output_g_base + oc_l2 * hw_out;
                            float *sum_oc_l2_base                   = sum_g_base + oc_l2 * hw_out;

                            PRAGMA_OMP_FOR_COLLAPSE_NOWAIT(2)
                            for (int64_t oc = 0; oc < out_channel_section; oc += OCBLK()) {
                                for (int64_t tile_l0 = 0; tile_l0 < wg_blocks; tile_l0++) {
                                    const float *raw_output_c_base = raw_out_cl2_base + oc * wg_blocks;
                                    const float32x4_t vbias        = vld1q_f32(bias_g_base + oc_l2 + oc);
                                    float *output_oc_base          = output_oc_l2_base + oc * hw_out;
                                    float *sum_oc_base             = sum_oc_l2_base + oc * hw_out;

                                    int64_t tile_id        = tile_l0 + tile_l2;
                                    const int64_t batch_id = tile_id / num_hw_blocks;

                                    const int64_t tile_hw_id = tile_id % num_hw_blocks;
                                    const int64_t tile_h_id  = tile_hw_id / num_w_blocks;
                                    const int64_t tile_w_id  = tile_hw_id % num_w_blocks;

                                    const int64_t oh = tile_h_id * WGB2F3_OBLK();
                                    const int64_t ow = tile_w_id * WGB2F3_OBLK();

                                    conv2d_n4cx_wgb2f3_postp_output_block_fp32(
                                        raw_output_c_base + tile_l0 * OCBLK(),
                                        vbias,
                                        output_oc_base + batch_id * output_b_stride + (oh * dst_w + ow) * OCBLK(),
                                        sum_oc_base + batch_id * output_b_stride + (oh * dst_w + ow) * OCBLK(),
                                        k_out_wg_set_local_offset,
                                        dst_w,
                                        std::min((int64_t)WGB2F3_OBLK(), dst_h - oh),
                                        std::min((int64_t)WGB2F3_OBLK(), dst_w - ow),
                                        cp.fuse_flag);
                                } // close loop over tile
                            } // close loop over oc(register)
                        } // close loop over oc(l2)
                    } // finish postp
                } // close loop over ic(l2)
            } // close loop over batch-dst_h-dst_w
        } // close loop over group
    }
    return ppl::common::RC_SUCCESS;
}

size_t conv2d_n4cx_wgb2f3_get_converted_filter_size_fp32(
    const int64_t channels,
    const int64_t num_output,
    const int64_t group)
{
    const int64_t ic_group             = channels / group;
    const int64_t oc_group             = num_output / group;
    const size_t converted_filter_size = group * WGB2F3_NSET() * CEIL4(oc_group) * CEIL4(ic_group) * sizeof(float) + LLC_CACHELINE_SIZE();
    return converted_filter_size;
}

void conv2d_n4cx_wgb2f3_convert_filter_fp32(
    const float *filter,
    float *converted_filter,
    float *aux_filter_buffer,
    const int64_t channels,
    const int64_t num_output,
    const int64_t group,
    const int64_t k_in_ch_section,
    const int64_t k_out_ch_section)
{
    const float32x4_t vhalves = vdupq_n_f32(0.5f);

    const int64_t ic_group = channels / group;
    const int64_t oc_group = num_output / group;
    const int64_t ic_g_pck = CEIL4(ic_group);
    const int64_t oc_g_pck = CEIL4(oc_group);

    const int64_t in_ch_section  = CEIL4(std::min(k_in_ch_section, ic_group));
    const int64_t out_ch_section = CEIL4(std::min(k_out_ch_section, oc_group));

    size_t filter_wg_set_offset = oc_g_pck * ic_g_pck;

    for (int64_t g = 0; g < group; g++) {
        const float *filter_g_base     = filter + g * oc_group * ic_group * 9;
        float *converted_filter_g_base = converted_filter + g * WGB2F3_NSET() * filter_wg_set_offset;

        // first pass
        // note: pack channels to 4c
        float *aux_filter = aux_filter_buffer;
        float g_ic_pck[9 * ICBLK()];
        for (int64_t oc = 0; oc < oc_group; oc++) {
            for (int64_t ic = 0; ic < ic_group; ic += ICBLK()) {
                const float *filter_base = filter_g_base + oc * ic_group * 9 + ic * 9;
                const int64_t icV        = std::min((int64_t)ICBLK(), ic_group - ic);

                for (int64_t lane_id = 0; lane_id < icV; lane_id++) {
                    for (int64_t kidx = 0; kidx < 9; kidx++) {
                        g_ic_pck[kidx * ICBLK() + lane_id] = (float)(filter_base[lane_id * 9 + kidx]);
                    }
                }
                for (int64_t kidx = 0; kidx < 9; kidx++) {
                    for (int64_t lane_id = icV; lane_id < ICBLK(); lane_id++) {
                        g_ic_pck[kidx * ICBLK() + lane_id] = 0.0f;
                    }
                }

                float32x4_t g[4];
                float32x4_t bg[12];

                // B * G[0][:]
                g[0] = vld1q_f32(&g_ic_pck[0 * ICBLK()]);
                g[1] = vld1q_f32(&g_ic_pck[1 * ICBLK()]);
                g[2] = vld1q_f32(&g_ic_pck[2 * ICBLK()]);

                bg[0] = g[0];
                bg[1] = g[1];
                bg[2] = g[2];
                bg[3] = vmulq_f32(g[0], vhalves);
                bg[4] = vmulq_f32(g[1], vhalves);
                bg[5] = vmulq_f32(g[2], vhalves);
                bg[6] = bg[3];
                bg[7] = bg[4];
                bg[8] = bg[5];

                // B * G[1][:]
                g[0] = vld1q_f32(&g_ic_pck[3 * ICBLK()]);
                g[1] = vld1q_f32(&g_ic_pck[4 * ICBLK()]);
                g[2] = vld1q_f32(&g_ic_pck[5 * ICBLK()]);

                bg[3] = vfmaq_f32(bg[3], g[0], vhalves);
                bg[4] = vfmaq_f32(bg[4], g[1], vhalves);
                bg[5] = vfmaq_f32(bg[5], g[2], vhalves);
                bg[6] = vfmsq_f32(bg[6], g[0], vhalves);
                bg[7] = vfmsq_f32(bg[7], g[1], vhalves);
                bg[8] = vfmsq_f32(bg[8], g[2], vhalves);

                // B * G[2][:]
                g[0] = vld1q_f32(&g_ic_pck[6 * ICBLK()]);
                g[1] = vld1q_f32(&g_ic_pck[7 * ICBLK()]);
                g[2] = vld1q_f32(&g_ic_pck[8 * ICBLK()]);

                bg[3]  = vfmaq_f32(bg[3], g[0], vhalves);
                bg[4]  = vfmaq_f32(bg[4], g[1], vhalves);
                bg[5]  = vfmaq_f32(bg[5], g[2], vhalves);
                bg[6]  = vfmaq_f32(bg[6], g[0], vhalves);
                bg[7]  = vfmaq_f32(bg[7], g[1], vhalves);
                bg[8]  = vfmaq_f32(bg[8], g[2], vhalves);
                bg[9]  = g[0];
                bg[10] = g[1];
                bg[11] = g[2];

                // B*G * B_t[0:1][:]
                g[0] = vmulq_f32(bg[0], vhalves);
                g[1] = vmulq_f32(bg[0], vhalves);
                g[2] = vmulq_f32(bg[3], vhalves);
                g[3] = vmulq_f32(bg[3], vhalves);
                g[0] = vfmaq_f32(g[0], bg[1], vhalves);
                g[1] = vfmsq_f32(g[1], bg[1], vhalves);
                g[2] = vfmaq_f32(g[2], bg[4], vhalves);
                g[3] = vfmsq_f32(g[3], bg[4], vhalves);
                g[0] = vfmaq_f32(g[0], bg[2], vhalves);
                g[1] = vfmaq_f32(g[1], bg[2], vhalves);
                g[2] = vfmaq_f32(g[2], bg[5], vhalves);
                g[3] = vfmaq_f32(g[3], bg[5], vhalves);

                vst1q_f32(aux_filter, bg[0]); // q0, q1
                vst1q_f32(aux_filter + 1 * filter_wg_set_offset, g[0]); // q24, q25
                vst1q_f32(aux_filter + 2 * filter_wg_set_offset, g[1]); // q26, q27
                vst1q_f32(aux_filter + 3 * filter_wg_set_offset, bg[2]); // q4, q5
                vst1q_f32(aux_filter + 4 * filter_wg_set_offset, bg[3]); // q6, q7
                vst1q_f32(aux_filter + 5 * filter_wg_set_offset, g[2]); // q28, q29
                vst1q_f32(aux_filter + 6 * filter_wg_set_offset, g[3]); // q30, q31
                vst1q_f32(aux_filter + 7 * filter_wg_set_offset, bg[5]); // q10, q11

                // B*G * B_t[2:3][:]
                g[0] = vmulq_f32(bg[6], vhalves);
                g[1] = vmulq_f32(bg[6], vhalves);
                g[2] = vmulq_f32(bg[9], vhalves);
                g[3] = vmulq_f32(bg[9], vhalves);
                g[0] = vfmaq_f32(g[0], bg[7], vhalves);
                g[1] = vfmsq_f32(g[1], bg[7], vhalves);
                g[2] = vfmaq_f32(g[2], bg[10], vhalves);
                g[3] = vfmsq_f32(g[3], bg[10], vhalves);
                g[0] = vfmaq_f32(g[0], bg[8], vhalves);
                g[1] = vfmaq_f32(g[1], bg[8], vhalves);
                g[2] = vfmaq_f32(g[2], bg[11], vhalves);
                g[3] = vfmaq_f32(g[3], bg[11], vhalves);

                vst1q_f32(aux_filter + 8 * filter_wg_set_offset, bg[6]);
                vst1q_f32(aux_filter + 9 * filter_wg_set_offset, g[0]);
                vst1q_f32(aux_filter + 10 * filter_wg_set_offset, g[1]);
                vst1q_f32(aux_filter + 11 * filter_wg_set_offset, bg[8]);
                vst1q_f32(aux_filter + 12 * filter_wg_set_offset, bg[9]);
                vst1q_f32(aux_filter + 13 * filter_wg_set_offset, g[2]);
                vst1q_f32(aux_filter + 14 * filter_wg_set_offset, g[3]);
                vst1q_f32(aux_filter + 15 * filter_wg_set_offset, bg[11]);

                aux_filter += ICBLK();
            }
        }

        // second pass
        // note: pack num_output to 8c
        for (int64_t set_id = 0; set_id < WGB2F3_NSET(); set_id++) {
            const float *aux_filter_base = aux_filter_buffer + set_id * filter_wg_set_offset;
            float *converted_filter_base = converted_filter_g_base + set_id * filter_wg_set_offset;

            for (int64_t i = 0; i < oc_group; i += out_ch_section) {
                for (int64_t p = 0; p < ic_g_pck; p += in_ch_section) {
                    int64_t m_l1 = std::min(oc_group - i, out_ch_section);
                    int64_t k_l1 = std::min(ic_g_pck - p, in_ch_section);
                    sgemm_n4cx_inner_blocking_8x4_fp32(
                        aux_filter_base + i * ic_g_pck + p,
                        converted_filter_base + i * CEIL4(ic_g_pck) + p * CEIL4(m_l1),
                        ic_g_pck,
                        m_l1,
                        k_l1);
                } // close loop over outer K blocks
            } // close loop over outer M blocks
        } // close loop over wg-sets
    }
}

bool conv2d_wgb2f3_fp32_offline_manager::is_supported()
{
    const conv2d_param &cp = param_;
    if (cp.group > 1) {
        const int64_t ic_group = cp.channels / cp.group;
        const int64_t oc_group = cp.num_output / cp.group;
        if (ic_group % ICBLK() != 0 || oc_group % OCBLK()) {
            return false;
        }
    }
    if (cp.kernel_h != 3 || cp.kernel_w != 3 ||
        cp.stride_h != 1 || cp.stride_w != 1 ||
        cp.dilation_h != 1 || cp.dilation_w != 1) {
        return false;
    }
    return true;
}

std::vector<int64_t>  conv2d_wgb2f3_fp32_offline_manager::get_schedule_param() const
{
    std::vector<int64_t> sp = { sched_param_.ic_seg, sched_param_.oc_seg, sched_param_.tile_seg };
    return sp;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_offline_manager::set_schedule_param(const std::vector<int64_t>& sp)
{
    if (sp.size() != 3) {
        return fast_init_schedule_param();
    }
    sched_param_.ic_seg = sp[0];
    sched_param_.oc_seg = sp[1];
    sched_param_.tile_seg = sp[2];
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_offline_manager::fast_init_schedule_param()
{
    sched_param_.oc_seg   = 44;
    sched_param_.ic_seg   = 128;
    sched_param_.tile_seg = 60;

    if (sched_param_.oc_seg != 44) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (sched_param_.ic_seg != 128) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (sched_param_.tile_seg != 60) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_offline_manager::pick_best_schedule_param(
    const ppl::common::TensorShape &src_shape,
    void *src,
    void *cvt_bias,
    const ppl::common::TensorShape &dst_shape,
    void *dst,
    bool tune_sp,
    double &run_time)
{
    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;

    if (src_shape.GetDimCount() < 4) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int64_t num_batch = src_shape.GetDim(0);
    const int64_t src_h     = src_shape.GetDim(2);
    const int64_t src_w     = src_shape.GetDim(3);

    auto conv_exe = conv2d_wgb2f3_fp32_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
    conv_exe.set_src(src);
    conv_exe.set_src_shape(&src_shape);
    conv_exe.set_dst(dst);
    conv_exe.set_dst_shape(&dst_shape);
    conv_exe.set_cvt_bias(cvt_bias);

    uint64_t cvt_filter_size = conv2d_n4cx_wgb2f3_get_converted_filter_size_fp32(
        channels, num_output, param_.group);
    float *cvt_filter      = (float *)allocator_->Alloc(cvt_filter_size);
    conv_exe.set_cvt_filter(cvt_filter);

    std::vector<int64_t> candidate_oc_blk_list   = {44};
    std::vector<int64_t> candidate_ic_blk_list   = {128};
    std::vector<int64_t> candidate_tile_blk_list = {60};
    if (tune_sp) {
        candidate_oc_blk_list   = {32, 44, 64,  128};
        candidate_ic_blk_list   = {32, 64, 128, 256};
        candidate_tile_blk_list = {32, 60, 128, 256};
    }

    size_t tmp_buf_size = 0;
    for (auto oc_seg : candidate_oc_blk_list) {
        for (auto ic_seg : candidate_ic_blk_list) {
            for (auto tile_seg : candidate_tile_blk_list) {
                conv_exe.sched_param_.oc_seg   = oc_seg;
                conv_exe.sched_param_.ic_seg   = ic_seg;
                conv_exe.sched_param_.tile_seg = tile_seg;
                conv_exe.prepare();

                const size_t new_size = conv_exe.cal_temp_buffer_size();
                if (new_size > tmp_buf_size) {
                    tmp_buf_size = new_size;
                }
            }
        }
    }
    float *tmp_buffer     = (float *)allocator_->Alloc(tmp_buf_size);
    conv_exe.set_temp_buffer(tmp_buffer);

    int64_t best_oc_blk   = 44;
    int64_t best_ic_blk   = 128;
    int64_t best_tile_blk = 60;
    int64_t best_run_time = std::numeric_limits<int64_t>::max();

    const int num_warmup_iter    = 1;
    const int num_benchmark_iter = 3;
    for (auto oc_seg : candidate_oc_blk_list) {
        for (auto ic_seg : candidate_ic_blk_list) {
            for (auto tile_seg : candidate_tile_blk_list) {
                conv_exe.sched_param_.oc_seg   = oc_seg;
                conv_exe.sched_param_.ic_seg   = ic_seg;
                conv_exe.sched_param_.tile_seg = tile_seg;
                conv_exe.adjust_schedule_param();

                for (int i = 0; i < num_warmup_iter; i++) {
                    conv_exe.execute();
                }

                auto begin_ts = std::chrono::system_clock::now();
                for (int i = 0; i < num_benchmark_iter; i++) {
                    conv_exe.execute();
                }
                auto end_ts = std::chrono::system_clock::now();

                int64_t elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - begin_ts).count();
                if (elapsed_time < best_run_time) {
                    best_oc_blk   = oc_seg;
                    best_ic_blk   = ic_seg;
                    best_tile_blk = tile_seg;
                    best_run_time = elapsed_time;
                }

                if (tile_seg >= num_batch * ((src_h + 1) / 2) * ((src_w + 1) / 2)) break;
            }
            if (ic_seg >= channels) break;
        }
        if (oc_seg >= num_output) break;
    }

    allocator_->Free(cvt_filter);
    allocator_->Free(tmp_buffer);

    sched_param_.oc_seg   = best_oc_blk;
    sched_param_.ic_seg   = best_ic_blk;
    sched_param_.tile_seg = best_tile_blk;
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    LOG(INFO) << "choose sp param oc: " << sched_param_.oc_seg;
    LOG(INFO) << "choose sp param ic: " << sched_param_.ic_seg;
    LOG(INFO) << "choose sp param tile: " << sched_param_.tile_seg;
    LOG(INFO) << "best run time: " << best_run_time / num_benchmark_iter / 1000 << " ms";
#endif
    run_time = (double)best_run_time / (double)num_benchmark_iter;
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_offline_manager::try_fuse(conv_fuse_flag_t fuse_type)
{
    return ((fuse_type | conv_fuse_flag::HSWISH) || (fuse_type | conv_fuse_flag::PRELU )) ?
        ppl::common::RC_UNSUPPORTED : ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_offline_manager::generate_cvt_weights_shapes(
    ppl::common::TensorShape &cvt_filter_shape,
    ppl::common::TensorShape &cvt_bias_shape)
{
    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;

    cvt_bias_size_ = CEIL4(num_output) * sizeof(float);
    cvt_bias_shape.SetDimCount(1);
    cvt_bias_shape.SetDim(0, cvt_bias_size_/sizeof(float));
    cvt_bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);

    cvt_filter_size_ = conv2d_n4cx_wgb2f3_get_converted_filter_size_fp32(
        channels, num_output, param_.group);
    cvt_filter_shape.SetDimCount(1);
    cvt_filter_shape.SetDim(0, cvt_filter_size_/sizeof(float));
    cvt_filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_wgb2f3_fp32_offline_manager::generate_cvt_weights(
    const void *filter,
    const void *bias,
    void* new_filter,
    void* new_bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;

    if (!bias && new_bias) {
        cvt_bias_ = new_bias;
    } else if (bias && new_bias) {
        cvt_bias_ = new_bias;
        int64_t padding_offset_bytes = num_output * sizeof(float);
        int64_t padding_bytes        = (CEIL4(num_output) - num_output) * sizeof(float);
        std::memcpy(cvt_bias_, bias, num_output * sizeof(float));
        std::memset((uint8_t *)cvt_bias_ + padding_offset_bytes, 0, padding_bytes);
    } else {
        cvt_bias_ = allocator_->Alloc(cvt_bias_size_);
        std::memset(cvt_bias_, 0, cvt_bias_size_);
        is_bias_owner_ = true;
    }

    cvt_filter_ = new_filter;

    const int64_t ic_group = channels / param_.group;
    const int64_t oc_group = num_output / param_.group;
    size_t buffer_size     = WGB2F3_NSET() * CEIL4(ic_group) * CEIL4(oc_group) * sizeof(float);
    float *aux_buffer      = (float *)allocator_->Alloc(buffer_size);
    conv2d_n4cx_wgb2f3_convert_filter_fp32(
        (const float *)filter,
        (float *)cvt_filter_,
        aux_buffer,
        channels,
        num_output,
        param_.group,
        sched_param_.ic_seg,
        sched_param_.oc_seg);
    allocator_->Free(aux_buffer);
    
    return ppl::common::RC_SUCCESS;
}

conv2d_runtime_executor *conv2d_wgb2f3_fp32_offline_manager::gen_executor()
{
    return new conv2d_wgb2f3_fp32_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

#undef CBLK
#undef ICBLK
#undef OCBLK

}}}}; // namespace ppl::kernel::arm_server::neon
