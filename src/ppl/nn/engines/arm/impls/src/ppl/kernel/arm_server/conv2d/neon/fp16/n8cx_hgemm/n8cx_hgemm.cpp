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

#include "ppl/kernel/arm_server/conv2d/neon/fp16/n8cx_hgemm/n8cx_hgemm.h"

#include <algorithm>
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define V_TRANSPOSE_FP16_8x8(v)                                                                               \
    do {                                                                                                      \
        float16x8x2_t vpf16[4];                                                                               \
        float32x4x2_t vpf32[4];                                                                               \
        vpf16[0] = vtrnq_f16(v[0], v[1]);                                                                     \
        vpf16[1] = vtrnq_f16(v[2], v[3]);                                                                     \
        vpf16[2] = vtrnq_f16(v[4], v[5]);                                                                     \
        vpf16[3] = vtrnq_f16(v[6], v[7]);                                                                     \
        vpf32[0] = vtrnq_f32((float32x4_t)vpf16[0].val[0], (float32x4_t)vpf16[1].val[0]);                     \
        vpf32[1] = vtrnq_f32((float32x4_t)vpf16[0].val[1], (float32x4_t)vpf16[1].val[1]);                     \
        vpf32[2] = vtrnq_f32((float32x4_t)vpf16[2].val[0], (float32x4_t)vpf16[3].val[0]);                     \
        vpf32[3] = vtrnq_f32((float32x4_t)vpf16[2].val[1], (float32x4_t)vpf16[3].val[1]);                     \
        v[0]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[2].val[0]));   \
        v[1]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[0]), vget_low_f32(vpf32[3].val[0]));   \
        v[2]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[2].val[1]));   \
        v[3]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[1]), vget_low_f32(vpf32[3].val[1]));   \
        v[4]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[2].val[0])); \
        v[5]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[0]), vget_high_f32(vpf32[3].val[0])); \
        v[6]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[2].val[1])); \
        v[7]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[1]), vget_high_f32(vpf32[3].val[1])); \
    } while (0)

void hgemm_n8cx_inner_blocking_8x8_fp16(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k)
{
    const float16x8_t vzeros = vdupq_n_f16(0.0f);

    for (int64_t i = 0; i < m; i += 8) {
        for (int64_t p = 0; p < k; p += 8) {
            int64_t m_l = std::min(m - i, (int64_t)8);
            int64_t k_l = std::min(k - p, (int64_t)8);

            float16x8_t v[8]; // 8 vec reg

            const __fp16 *a_ptr = a + i * lda + p;

            if (k_l == 8) {
                if (m_l == 8) {
                    v[0] = vld1q_f16(a_ptr + 0 * lda);
                    v[1] = vld1q_f16(a_ptr + 1 * lda);
                    v[2] = vld1q_f16(a_ptr + 2 * lda);
                    v[3] = vld1q_f16(a_ptr + 3 * lda);
                    v[4] = vld1q_f16(a_ptr + 4 * lda);
                    v[5] = vld1q_f16(a_ptr + 5 * lda);
                    v[6] = vld1q_f16(a_ptr + 6 * lda);
                    v[7] = vld1q_f16(a_ptr + 7 * lda);
                } else {
                    for (int64_t id = 0; id < m_l; id++) {
                        v[id] = vld1q_f16(a_ptr + id * lda);
                    }
                    for (int64_t id = m_l; id < 8; id++) {
                        v[id] = vzeros;
                    }
                }

                V_TRANSPOSE_FP16_8x8(v);

                vst1q_f16(converted_a + 0, v[0]);
                vst1q_f16(converted_a + 8, v[1]);
                vst1q_f16(converted_a + 16, v[2]);
                vst1q_f16(converted_a + 24, v[3]);
                vst1q_f16(converted_a + 32, v[4]);
                vst1q_f16(converted_a + 40, v[5]);
                vst1q_f16(converted_a + 48, v[6]);
                vst1q_f16(converted_a + 56, v[7]);
            } else {
                for (int64_t ii = 0; ii < m_l; ii++) {
                    for (int64_t pp = 0; pp < k_l; pp++) {
                        converted_a[pp * 8 + ii] = a_ptr[ii * lda + pp];
                    }
                }
                for (int64_t ii = m_l; ii < 8; ii++) {
                    for (int64_t pp = 0; pp < k_l; pp++) {
                        converted_a[pp * 8 + ii] = 0.0f;
                    }
                }
                for (int64_t pp = k_l; pp < 8; pp++) {
                    vst1q_f16(converted_a + pp * 8, vzeros);
                }
            }

            converted_a += 64;
        } // close loop over inner k blocks
    } // close loop over inner m blocks
}

void hgemm_n8cx_inner_blocking_16x8_fp16(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k)
{
    const float16x8_t vzeros = vdupq_n_f16(0.0f);

    int64_t i = 0;
    for (; i < m; i += 16) {
        for (int64_t p = 0; p < k; p += 8) {
            int64_t m_l = std::min(m - i, (int64_t)16);
            int64_t k_l = std::min(k - p, (int64_t)8);

            float16x8_t v[16]; // 16 vec reg
            float16x8x2_t vpf16[4]; // 8 vec reg
            float32x4x2_t vpf32[4]; // 8 vec reg

            const __fp16 *a_ptr = a + i * lda + p;

            int64_t cvt_a_offset;
            if (k_l == 8 && m_l == 16) {
                v[0] = vld1q_f16(a_ptr + 0 * lda);
                v[1] = vld1q_f16(a_ptr + 1 * lda);
                v[2] = vld1q_f16(a_ptr + 2 * lda);
                v[3] = vld1q_f16(a_ptr + 3 * lda);
                v[4] = vld1q_f16(a_ptr + 4 * lda);
                v[5] = vld1q_f16(a_ptr + 5 * lda);
                v[6] = vld1q_f16(a_ptr + 6 * lda);
                v[7] = vld1q_f16(a_ptr + 7 * lda);

                v[8]  = vld1q_f16(a_ptr + 8 * lda);
                v[9]  = vld1q_f16(a_ptr + 9 * lda);
                v[10] = vld1q_f16(a_ptr + 10 * lda);
                v[11] = vld1q_f16(a_ptr + 11 * lda);
                v[12] = vld1q_f16(a_ptr + 12 * lda);
                v[13] = vld1q_f16(a_ptr + 13 * lda);
                v[14] = vld1q_f16(a_ptr + 14 * lda);
                v[15] = vld1q_f16(a_ptr + 15 * lda);

                vpf16[0] = vtrnq_f16(v[0], v[1]);
                vpf16[1] = vtrnq_f16(v[2], v[3]);
                vpf16[2] = vtrnq_f16(v[4], v[5]);
                vpf16[3] = vtrnq_f16(v[6], v[7]);
                vpf32[0] = vtrnq_f32((float32x4_t)vpf16[0].val[0], (float32x4_t)vpf16[1].val[0]);
                vpf32[1] = vtrnq_f32((float32x4_t)vpf16[0].val[1], (float32x4_t)vpf16[1].val[1]);
                vpf32[2] = vtrnq_f32((float32x4_t)vpf16[2].val[0], (float32x4_t)vpf16[3].val[0]);
                vpf32[3] = vtrnq_f32((float32x4_t)vpf16[2].val[1], (float32x4_t)vpf16[3].val[1]);
                v[0]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[2].val[0]));
                v[1]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[0]), vget_low_f32(vpf32[3].val[0]));
                v[2]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[2].val[1]));
                v[3]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[1]), vget_low_f32(vpf32[3].val[1]));
                v[4]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[2].val[0]));
                v[5]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[0]), vget_high_f32(vpf32[3].val[0]));
                v[6]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[2].val[1]));
                v[7]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[1]), vget_high_f32(vpf32[3].val[1]));

                vst1q_f16(converted_a + 0, v[0]);
                vst1q_f16(converted_a + 16, v[1]);
                vst1q_f16(converted_a + 32, v[2]);
                vst1q_f16(converted_a + 48, v[3]);
                vst1q_f16(converted_a + 64, v[4]);
                vst1q_f16(converted_a + 80, v[5]);
                vst1q_f16(converted_a + 96, v[6]);
                vst1q_f16(converted_a + 112, v[7]);

                vpf16[0] = vtrnq_f16(v[8], v[9]);
                vpf16[1] = vtrnq_f16(v[10], v[11]);
                vpf16[2] = vtrnq_f16(v[12], v[13]);
                vpf16[3] = vtrnq_f16(v[14], v[15]);
                vpf32[0] = vtrnq_f32((float32x4_t)vpf16[0].val[0], (float32x4_t)vpf16[1].val[0]);
                vpf32[1] = vtrnq_f32((float32x4_t)vpf16[0].val[1], (float32x4_t)vpf16[1].val[1]);
                vpf32[2] = vtrnq_f32((float32x4_t)vpf16[2].val[0], (float32x4_t)vpf16[3].val[0]);
                vpf32[3] = vtrnq_f32((float32x4_t)vpf16[2].val[1], (float32x4_t)vpf16[3].val[1]);
                v[8]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[2].val[0]));
                v[9]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[0]), vget_low_f32(vpf32[3].val[0]));
                v[10]    = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[2].val[1]));
                v[11]    = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[1]), vget_low_f32(vpf32[3].val[1]));
                v[12]    = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[2].val[0]));
                v[13]    = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[0]), vget_high_f32(vpf32[3].val[0]));
                v[14]    = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[2].val[1]));
                v[15]    = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[1]), vget_high_f32(vpf32[3].val[1]));

                vst1q_f16(converted_a + 8, v[8]);
                vst1q_f16(converted_a + 24, v[9]);
                vst1q_f16(converted_a + 40, v[10]);
                vst1q_f16(converted_a + 56, v[11]);
                vst1q_f16(converted_a + 72, v[12]);
                vst1q_f16(converted_a + 88, v[13]);
                vst1q_f16(converted_a + 104, v[14]);
                vst1q_f16(converted_a + 120, v[15]);

                cvt_a_offset = 128;
            } else if (k_l == 8 && m_l == 8) {
                v[0] = vld1q_f16(a_ptr + 0 * lda);
                v[1] = vld1q_f16(a_ptr + 1 * lda);
                v[2] = vld1q_f16(a_ptr + 2 * lda);
                v[3] = vld1q_f16(a_ptr + 3 * lda);
                v[4] = vld1q_f16(a_ptr + 4 * lda);
                v[5] = vld1q_f16(a_ptr + 5 * lda);
                v[6] = vld1q_f16(a_ptr + 6 * lda);
                v[7] = vld1q_f16(a_ptr + 7 * lda);

                vpf16[0] = vtrnq_f16(v[0], v[1]);
                vpf16[1] = vtrnq_f16(v[2], v[3]);
                vpf16[2] = vtrnq_f16(v[4], v[5]);
                vpf16[3] = vtrnq_f16(v[6], v[7]);
                vpf32[0] = vtrnq_f32((float32x4_t)vpf16[0].val[0], (float32x4_t)vpf16[1].val[0]);
                vpf32[1] = vtrnq_f32((float32x4_t)vpf16[0].val[1], (float32x4_t)vpf16[1].val[1]);
                vpf32[2] = vtrnq_f32((float32x4_t)vpf16[2].val[0], (float32x4_t)vpf16[3].val[0]);
                vpf32[3] = vtrnq_f32((float32x4_t)vpf16[2].val[1], (float32x4_t)vpf16[3].val[1]);
                v[0]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[0]), vget_low_f32(vpf32[2].val[0]));
                v[1]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[0]), vget_low_f32(vpf32[3].val[0]));
                v[2]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[0].val[1]), vget_low_f32(vpf32[2].val[1]));
                v[3]     = (float16x8_t)vcombine_f32(vget_low_f32(vpf32[1].val[1]), vget_low_f32(vpf32[3].val[1]));
                v[4]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[0]), vget_high_f32(vpf32[2].val[0]));
                v[5]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[0]), vget_high_f32(vpf32[3].val[0]));
                v[6]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[0].val[1]), vget_high_f32(vpf32[2].val[1]));
                v[7]     = (float16x8_t)vcombine_f32(vget_high_f32(vpf32[1].val[1]), vget_high_f32(vpf32[3].val[1]));

                vst1q_f16(converted_a + 0, v[0]);
                vst1q_f16(converted_a + 8, v[1]);
                vst1q_f16(converted_a + 16, v[2]);
                vst1q_f16(converted_a + 24, v[3]);
                vst1q_f16(converted_a + 32, v[4]);
                vst1q_f16(converted_a + 40, v[5]);
                vst1q_f16(converted_a + 48, v[6]);
                vst1q_f16(converted_a + 56, v[7]);

                cvt_a_offset = 64;
            } else {
                const int64_t m_l_pck = CEIL8(m_l);
                for (int64_t pp = 0; pp < k_l; pp++) {
                    for (int64_t ii = 0; ii < m_l; ii++) {
                        converted_a[pp * m_l_pck + ii] = a_ptr[ii * lda + pp];
                    }
                    for (int64_t ii = m_l; ii < m_l_pck; ii++) {
                        converted_a[pp * m_l_pck + ii] = 0.0f;
                    }
                }
                for (int64_t pp = k_l; pp < 8; pp++) {
                    if (m_l_pck == 8) {
                        vst1q_f16(converted_a + pp * 8, vzeros);
                    } else if (m_l_pck == 16) {
                        vst1q_f16(converted_a + pp * 16, vzeros);
                        vst1q_f16(converted_a + pp * 16 + 8, vzeros);
                    }
                }

                cvt_a_offset = m_l_pck * 8;
            }

            converted_a += cvt_a_offset;
        } // close loop over inner k blocks
    } // close loop over inner m blocks
}

template <>
void hgemm_n8cx_blocking_fp16<N8cxHgemmBlockingOrd::M_N_K>(
    const __fp16 *a,
    __fp16 *converted_a,
    const int64_t lda,
    const int64_t m,
    const int64_t k,
    const int64_t m_block1,
    const int64_t k_block1)
{
    for (int64_t i = 0; i < m; i += m_block1) {
        for (int64_t p = 0; p < k; p += k_block1) {
            int64_t m_l1 = std::min(m - i, m_block1);
            int64_t k_l1 = std::min(k - p, k_block1);

            hgemm_n8cx_inner_blocking_8x8_fp16(
                a + i * lda + p,
                converted_a + i * CEIL8(k) + p * CEIL8(m_l1),
                lda,
                m_l1,
                k_l1);

        } // close loop over outer K blocks
    } // close loop over outer M blocks
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
