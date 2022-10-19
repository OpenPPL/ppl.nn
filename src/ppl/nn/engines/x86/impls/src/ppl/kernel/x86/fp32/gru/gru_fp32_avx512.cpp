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

#include <math.h>
#include <string.h>
#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gru.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/common/avx512_tools.h"
#include "ppl/kernel/x86/common/math_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

static inline float sigmoidf(const float x)
{
    return 1.0f / (1.0f + expf(-x));
}

ppl::common::RetCode gru_fp32_avx512(
    const ppl::nn::TensorShape *X_shape,
    const float *X,
    const float **W_weight,
    const float **Rzr_weight,
    const float **Rh_weight,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    bool has_packed_W,
    bool has_packed_Rzr,
    bool has_packed_Rh,
    void *temp_buffer,
    float *Y,
    float *Y_h)
{
    if (!Y && !Y_h) {
        return ppl::common::RC_SUCCESS;
    }
    const int64_t simd_w        = 16;
    const __m512 vone           = _mm512_set1_ps(1.0f);
    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;
    const int64_t seq_len       = X_shape->GetDim(0);
    const int64_t batch         = X_shape->GetDim(1);
    const int64_t input_size    = X_shape->GetDim(2);
    float *Yh_buf               = Y_h;
    float *temp_buffer_fp32     = reinterpret_cast<float *>(temp_buffer);
    if (!Yh_buf) {
        Yh_buf = temp_buffer_fp32;
        temp_buffer_fp32 += num_direction * batch * hidden_size;
    }
    float *gate_buf       = temp_buffer_fp32;
    float *gate_extra_buf = gate_buf + batch * rnn_num_gate::GRU * hidden_size;
    for (int64_t nd = 0; nd < num_direction; ++nd) {
        const bool is_reverse = nd || (direction == rnn_direction::REVERSE);
        float *nd_Yh          = Yh_buf + nd * batch * hidden_size;
        float *nd_Y           = Y + nd * batch * hidden_size;
        // const float *nd_W      = W_weight + nd * rnn_num_gate::GRU * hidden_size * input_size;
        // const float *nd_R      = R_weight + nd * rnn_num_gate::GRU * hidden_size * hidden_size;
        const float *nd_W     = W_weight[nd];
        const float *nd_Rzr   = Rzr_weight[nd];
        const float *nd_Rh    = Rh_weight[nd];

        const float *nd_Wb     = bias ? bias + nd * 2 * rnn_num_gate::GRU * hidden_size : nullptr; // [Wb[zrh], Rb[zrh]] * directions
        const float *nd_Rb     = bias ? nd_Wb + rnn_num_gate::GRU * hidden_size : nullptr;
        const float *nd_init_h = initial_h ? initial_h + nd * batch * hidden_size : nullptr;

        for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            const int64_t mapped_seq_index = is_reverse ? (seq_len - seq_idx - 1) : seq_idx;
            const bool is_first_seq        = seq_idx == 0;
            const float *sX                = X + mapped_seq_index * batch * input_size;
            const float *Y_h_prev          = is_first_seq ? nd_init_h : nd_Yh;
            float *sY                      = nd_Y + mapped_seq_index * num_direction * batch * hidden_size;

            gemm_fp32_avx512( // Xt*(Wz^T) + Wbz ; Xt*(Wr^T) + Wbr ; Xt*(Wh^T) + Wbh
                sX,
                nd_W,
                nd_Wb,
                nullptr,
                gemm_m_type::NOTRANS,
                has_packed_W ? gemm_m_type::PACKED : gemm_m_type::TRANS,
                nd_Wb ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY,
                gemm_m_type::EMPTY,
                batch,
                rnn_num_gate::GRU * hidden_size,
                input_size,
                input_size,
                input_size,
                rnn_num_gate::GRU * hidden_size,
                0,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                gemm_post::NONE,
                gate_buf);
            const float thisK = !Y_h_prev ? 0 : hidden_size; // some hack
            gemm_fp32_avx512( // Ht-1*(Rz^T) + Rbz; Ht-1*(Rr^T) + Rbr
                Y_h_prev,
                nd_Rzr,
                nd_Rb,
                nullptr,
                gemm_m_type::NOTRANS,
                has_packed_Rzr ? gemm_m_type::PACKED : gemm_m_type::TRANS,
                nd_Rb ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY,
                gemm_m_type::EMPTY,
                batch,
                (rnn_num_gate::GRU - 1) * hidden_size,
                thisK,
                hidden_size,
                hidden_size,
                rnn_num_gate::GRU * hidden_size,
                0,
                1.0f,
                1.0f,
                1.0f,
                0.0f,
                gemm_post::NONE,
                gate_buf);

            // const float *nd_Rh  = nd_R ? nd_R + 2 * hidden_size * hidden_size : nullptr; // nd_R[zrh]->nd_Rh
            const float *nd_Rbh = nd_Rb ? nd_Rb + 2 * hidden_size : nullptr; // nd_Rb[zrh]->nd_Rbh
            gemm_fp32_avx512( //  (Ht-1*(Rh^T) + Rbh)
                Y_h_prev,
                nd_Rh,
                nd_Rbh,
                nullptr,
                gemm_m_type::NOTRANS,
                has_packed_Rh ? gemm_m_type::PACKED : gemm_m_type::TRANS,
                nd_Rbh ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY,
                gemm_m_type::EMPTY,
                batch,
                hidden_size,
                thisK,
                hidden_size,
                hidden_size,
                hidden_size,
                hidden_size,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                gemm_post::NONE,
                gate_extra_buf);

            if (is_first_seq && !Y_h_prev) {
                Y_h_prev = nd_Yh; // preprocess Y_h_prev
                memset32_avx(nd_Yh, 0, batch * hidden_size);
            }

            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t b = 0; b < batch; ++b) {
                if (!sequence_lens || seq_idx < sequence_lens[b]) {
                    const float *gZ    = gate_buf + b * rnn_num_gate::GRU * hidden_size;
                    const float *gR    = gZ + hidden_size;
                    const float *gH    = gR + hidden_size;
                    const float *gE    = gate_extra_buf + b * hidden_size;
                    const float *Hprev = Y_h_prev + b * hidden_size;
                    float *Ht          = nd_Yh + b * hidden_size;
                    float *Yt          = sY + b * hidden_size;
                    int64_t h          = 0;
                    for (; h <= hidden_size - simd_w; h += simd_w) {
                        const __m512 zt   = _avx512_sigmoid_ps(_mm512_loadu_ps(gZ + h));
                        const __m512 rt   = _avx512_sigmoid_ps(_mm512_loadu_ps(gR + h));
                        const __m512 et   = _mm512_loadu_ps(gE + h);
                        __m512 ht         = _avx512_tanh_ps(rt * et + _mm512_loadu_ps(gH + h));
                        const __m512 hpre = _mm512_loadu_ps(Hprev + h);
                        const __m512 res  = (vone - zt) * ht + zt * hpre;
                        _mm512_storeu_ps(Ht + h, res);
                    }
                    for (; h < hidden_size; ++h) {
                        const float zt = sigmoidf(gZ[h]);
                        const float rt = sigmoidf(gR[h]);
                        const float ht = ::tanhf(rt * gE[h] + gH[h]);
                        Ht[h]          = (1 - zt) * ht + zt * Hprev[h];
                    }
                    if (Y) {
                        memcpy32_avx(Yt, Ht, hidden_size);
                    }
                } else { // pass through the initial_h, initial_c
                    const float *Hprev = Y_h_prev + b * hidden_size;
                    float *Ht          = nd_Yh + b * hidden_size;
                    memcpy32_avx(Ht, Hprev, hidden_size);
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
