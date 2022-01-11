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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/utils/conv2d_utils_fp32.h"

#include <arm_neon.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "ppl/common/arm/sysinfo.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define CBLK()  4
#define ICBLK() CBLK()
#define OCBLK() CBLK()

void conv2d_n4cx_load_group_fp32(
    const float *input_b_base,
    float *input_gbuf_g_base,
    const int64_t hw_in,
    const int64_t ic_group,
    const int64_t gid_global,
    const int64_t gid_local)
{
    const int64_t ic_group_align = FLOOR4(ic_group);

    int64_t channel_g_base = gid_global * ic_group;

    PRAGMA_OMP_FOR_NOWAIT()
    for (int64_t ic = 0; ic < ic_group_align; ic += ICBLK()) {
        int64_t channel_base = channel_g_base + ic;
        for (int64_t idx = 0; idx < hw_in; idx++) {
            const float *input_base = input_b_base + idx * ICBLK();
            float *input_gbuf_base  = input_gbuf_g_base + ic * hw_in + idx * ICBLK();

            // TODO: vectorization
            for (int64_t lane = 0; lane < ICBLK(); lane++) {
                int src_idx           = FLOOR4(channel_base + lane) * hw_in + (channel_base + lane) % ICBLK();
                input_gbuf_base[lane] = input_base[src_idx];
            }
        }
    }
    const int64_t ic_tail = ic_group - ic_group_align;

    PRAGMA_OMP_SINGLE_NOWAIT()
    if (ic_tail) {
        int64_t channel_base = channel_g_base + ic_group_align;
        for (int64_t idx = 0; idx < hw_in; idx++) {
            const float *input_base = input_b_base + idx * ICBLK();
            float *input_gbuf_base  = input_gbuf_g_base + ic_group_align * hw_in + idx * ICBLK();

            // TODO: vectorization
            for (int64_t lane = 0; lane < ic_tail; lane++) {
                int src_idx           = FLOOR4(channel_base + lane) * hw_in + (channel_base + lane) % ICBLK();
                input_gbuf_base[lane] = input_base[src_idx];
            }
            for (int64_t lane = ic_tail; lane < ICBLK(); lane++) {
                input_gbuf_base[lane] = 0.0f;
            }
        }
    }
}

void conv2d_n4cx_store_group_fp32(
    const float *output_gbuf_g_base,
    float *output,
    float *sum,
    const int64_t hw_out,
    const int64_t oc_group,
    const int64_t gid_global,
    const int64_t gid_local,
    const int64_t fuse_flag)
{
    const int64_t global_oc_start = gid_global * oc_group;
    const int64_t global_oc_end   = global_oc_start + oc_group;

    int64_t global_oc_inner_start = CEIL4(global_oc_start);
    int64_t global_oc_inner_end   = FLOOR4(global_oc_end);

    global_oc_inner_start = std::min(global_oc_inner_start, global_oc_end);
    global_oc_inner_end   = std::max(global_oc_inner_end, global_oc_inner_start);

    const int64_t oc_head = global_oc_inner_start - global_oc_start;

    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vsix  = vdupq_n_f32(6.0f);

    PRAGMA_OMP_SINGLE_NOWAIT()
    if (oc_head) {
        const int64_t global_oc_base        = FLOOR4(global_oc_start);
        const int64_t global_oc_ofs_in_lane = global_oc_start - global_oc_base;
        for (int64_t idx = 0; idx < hw_out; idx++) {
            for (int64_t lane = 0; lane < oc_head; lane++) {
                float tmp_result = output_gbuf_g_base[idx * OCBLK() + lane];
                if (fuse_flag & conv_fuse_flag::SUM) {
                    tmp_result += sum[global_oc_base * hw_out + idx * OCBLK() + global_oc_ofs_in_lane + lane];
                }
                if (fuse_flag & conv_fuse_flag::RELU) {
                    tmp_result = std::max(tmp_result, 0.0f);
                }
                if (fuse_flag & conv_fuse_flag::RELU6) {
                    tmp_result = std::min(tmp_result, 6.0f);
                }
                output[global_oc_base * hw_out + idx * OCBLK() + global_oc_ofs_in_lane + lane] = tmp_result;
            }
        }
    }

    PRAGMA_OMP_FOR_NOWAIT()
    for (int64_t global_oc = global_oc_inner_start; global_oc < global_oc_inner_end; global_oc += OCBLK()) {
        for (int64_t idx = 0; idx < hw_out; idx++) {
            for (int64_t lane = 0; lane < OCBLK(); lane++) {
                int64_t channel_in_group                          = global_oc - global_oc_start + lane;
                int64_t buffer_idx                                = FLOOR4(channel_in_group) * hw_out + idx * OCBLK() + channel_in_group % OCBLK();
                output[global_oc * hw_out + idx * OCBLK() + lane] = output_gbuf_g_base[buffer_idx];
            }
            if (fuse_flag) {
                float32x4_t vout = vld1q_f32(output + global_oc * hw_out + idx * OCBLK());
                if (fuse_flag & conv_fuse_flag::SUM) {
                    vout = vaddq_f32(vout, vld1q_f32(sum + global_oc * hw_out + idx * OCBLK()));
                }
                if (fuse_flag & conv_fuse_flag::RELU) {
                    vout = vmaxq_f32(vout, vzero);
                }
                if (fuse_flag & conv_fuse_flag::RELU6) {
                    vout = vminq_f32(vout, vsix);
                }
                vst1q_f32(output + global_oc * hw_out + idx * OCBLK(), vout);
            }
        }
    }

    const int64_t oc_tail = global_oc_end - global_oc_inner_end;

    PRAGMA_OMP_SINGLE_NOWAIT()
    if (oc_tail) {
        const int64_t global_oc_base = global_oc_inner_end;
        const int64_t group_oc_ofs   = global_oc_inner_end - global_oc_start;
        for (int64_t idx = 0; idx < hw_out; idx++) {
            for (int64_t lane = 0; lane < oc_tail; lane++) {
                int64_t channel_in_group = group_oc_ofs + lane;
                int64_t buffer_idx       = FLOOR4(channel_in_group) * hw_out + idx * OCBLK() + channel_in_group % OCBLK();
                float tmp_result         = output_gbuf_g_base[buffer_idx];
                if (fuse_flag & conv_fuse_flag::SUM) {
                    tmp_result += sum[global_oc_base * hw_out + idx * OCBLK() + lane];
                }
                if (fuse_flag & conv_fuse_flag::RELU) {
                    tmp_result = std::max(tmp_result, 0.0f);
                }
                if (fuse_flag & conv_fuse_flag::RELU6) {
                    tmp_result = std::min(tmp_result, 6.0f);
                }
                output[global_oc_base * hw_out + idx * OCBLK() + lane] = tmp_result;
            }
        }
    }
}

#undef CBLK
#undef ICBLK
#undef OCBLK

}}}}; // namespace ppl::kernel::arm_server::neon
